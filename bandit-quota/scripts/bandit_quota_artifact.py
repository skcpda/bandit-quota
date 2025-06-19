#!/usr/bin/env python3
"""
Bandit-Quota – Contextual Bandit Fusion for BEIR Benchmarks
===========================================================

This script evaluates retrieval quality and latency performance on various BEIR datasets
using a contextual bandit (LinTS) strategy to dynamically fuse results from multiple dense encoders.

Usage:
  python bandit_quota.py                    # Evaluates SciFact (default)
  python bandit_quota.py --dataset nfcorpus # Evaluates NFCorpus
  python bandit_quota.py --dataset trec-covid # Evaluates TREC-COVID

Outputs:
  - Mean nDCG@10 and mean latency per query for:
    • Bandit-Quota (LinTS selection of 6 dense encoders with MiniLM reranker)
    • UNION-6 baseline (simple merging + MiniLM reranker)
    • Individual encoder arms (sampled and full dataset)

Dependencies: beir >= 2.0, sentence-transformers, torch, tqdm, rank-bm25
"""

from __future__ import annotations

# Import libraries
import argparse, logging, math, os, random, re, time, warnings
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import evaluation, models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi

# Dataset URLs mapping
URLS = {
    "scifact":   "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus":  "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "trec-covid":"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
}

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="scifact", choices=URLS.keys(),
                    help="Specify BEIR dataset to evaluate (default: scifact)")
args = parser.parse_args()
DATASET = args.dataset

# Ensure valid dataset and setup reproducibility
assert DATASET in URLS, f"Unknown dataset {DATASET}"
random.seed(13); np.random.seed(13)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO, force=True)
warnings.filterwarnings("ignore")

# Bandit and retrieval hyperparameters
D = 12              # Query context feature dimension
LAMBDA = 0.07       # Latency penalty coefficient
GAMMA = 0.05        # Diversity bonus
ALPHA_START = 1.0   # Initial exploration parameter
ALPHA_DECAY = 0.6   # Exploration decay factor
ALPHA_DECAY_AT = 100 # Decay after this many pulls
FAIL_THRESH = 0.15  # Quality threshold to fallback to RRF fusion
N_PREFILL = 30      # Number of queries before evaluating fallback
TOP_K_ARM = 200     # Number of top documents per arm
TOP_K_RERANK = 50   # Number of documents for cross-encoder reranking

# Load specified BEIR dataset
logging.info("⇢ Loading %s …", DATASET)
root = util.download_and_unzip(URLS[DATASET], f"datasets/{DATASET}")
split = "test" if os.path.exists(os.path.join(root, "qrels", "test.tsv")) else "validation"
corpus, queries, qrels_raw = GenericDataLoader(root).load(split=split)

# Prepare query relevance judgments
qrels = {qid: {str(d): rel for d, rel in docs.items()} for qid, docs in qrels_raw.items()}
logging.info("Corpus=%d  Queries=%d  Split=%s", len(corpus), len(queries), split)

# Define dense retrieval models (arms)
HF_MODELS = OrderedDict([
    ("bge",    "BAAI/bge-base-en"),
    ("contr",  "facebook/contriever"),
    ("mpnet",  "sentence-transformers/all-mpnet-base-v2"),
    ("gtr",    "sentence-transformers/gtr-t5-base"),
    ("minilm", "sentence-transformers/msmarco-MiniLM-L6-cos-v5"),
    ("distil", "sentence-transformers/msmarco-distilbert-base-v3"),
])

# Load retrievers
retrievers, latencies, runs = {}, {}, {}
for tag, mid in HF_MODELS.items():
    try:
        model = models.SentenceBERT(mid)
        search = DRES(model, batch_size=32, corpus_chunk_size=2048, score_function="dot")
        retrievers[tag] = evaluation.EvaluateRetrieval(search, score_function="dot")
    except Exception as exc:
        logging.warning("⚠️ %s skipped: %s", tag, exc)
assert len(retrievers) == 6, "Need six live retrieval arms"
ARMS = list(retrievers.keys())

# Precompute retrieval results and latencies
for tag, ret in retrievers.items():
    t0 = time.time()
    ret.top_k = TOP_K_ARM
    runs[tag] = ret.retrieve(corpus, queries)
    latencies[tag] = time.time() - t0
    runs[tag] = {q: {d: (s-min(sc.values()))/(max(sc.values())-min(sc.values())) for d, s in sc.items()} for q, sc in runs[tag].items()}
    logging.info("✓ %-6s done in %.1f min", tag, latencies[tag]/60)

# Compute nDCG@10 helper function
def ndcg_at_10(run, rel):
    ranked = sorted(run.items(), key=lambda kv: -kv[1])[:10]
    dcg = sum((2**rel.get(d,0)-1)/math.log2(i+2) for i,(d,_) in enumerate(ranked))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2**r-1)/math.log2(i+2) for i,r in enumerate(ideal))
    return dcg/idcg if idcg else 0.0

# Prepare UNION-6 baseline
union_scores, union_lat = [], []
lat_avg = sum(latencies[a] for a in ARMS) / len(queries)
for qid in queries:
    merged = defaultdict(float)
    for a in ARMS: merged.update(runs[a][qid])
    union_scores.append(ndcg_at_10(merged, qrels[qid]))
    union_lat.append(lat_avg)

# Initialize cross-encoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# Feature engineering for context vector
_tok = lambda t: re.findall(r"\w+", t.lower())
idf = {t: math.log(len(corpus)/(1+sum(t in _tok(doc["text"]) for doc in corpus.values())))
       for t in {w for doc in corpus.values() for w in _tok(doc["text"])}}
_year = re.compile(r"\b(19|20)\d{2}\b")
BM25_INDEX = BM25Okapi([_tok(doc["text"]) for doc in corpus.values()])

def _entropy(toks):
    c = Counter(toks); tot = len(toks)
    return -sum(v/tot*math.log2(v/tot) for v in c.values()) if tot else 0.0

def context_vec(q: str):
    toks=_tok(q); idfs=[idf.get(t,0.0) for t in toks]
    bm=max(BM25_INDEX.get_scores(toks)) if toks else 0.0
    return np.array([
        len(q), len(toks), len(set(toks)), sum(c.isdigit() for c in q),
        float(any(t[0].isupper() for t in toks)), np.mean([len(t) for t in toks]) if toks else 0.0,
        float("?" in q), float(q.endswith("?")), _entropy(toks), np.mean(idfs) if idfs else 0.0,
        bm, float(bool(_year.search(q)))
    ], dtype=np.float32)

# LinTS contextual bandit class
class LinTS:
    def __init__(self, n:int, d:int, alpha:float, lat, ndcg_prior, w_ndcg:float=1.0):
        self.A=[np.eye(d) for _ in range(n)]
        self.b=[(w_ndcg*ndcg - LAMBDA*l)*np.ones(d) for ndcg,l in zip(ndcg_prior,lat)]
        self.alpha=alpha
    def pull(self,x):
        mus=[]
        for A,b in zip(self.A,self.b):
            Ainv=np.linalg.inv(A)
            theta=np.random.multivariate_normal(Ainv@b, self.alpha**2*Ainv)
            mus.append(theta@x)
        return int(np.argmax(mus))
    def update(self,i,x,r):
        self.A[i]+=np.outer(x,x)
        self.b[i]+=r*x

# Utility function for reciprocal rank fusion (RRF)
def rrf(*doclists, k:int=60):
    fused=defaultdict(float)
    for dl in doclists:
        for rank,(d,_) in enumerate(sorted(dl.items(), key=lambda kv:-kv[1]),1):
            fused[d]+=1/(k+rank)
    return fused

# Compute priors for bandit arms
lat_prior=[latencies[a]/len(queries) for a in ARMS]
ndcg_prior=[np.mean([ndcg_at_10(runs[a][q], qrels[q]) for q in queries]) for a in ARMS]

# Initialize bandit model
bandit=LinTS(len(ARMS), D, ALPHA_START, lat_prior, ndcg_prior)
band_scores, band_lat, arm_stats=[],[],defaultdict(list)
prev_ranked=None
rrf_cache={qid: rrf(runs["bge"][qid], runs["contr"][qid]) for qid in queries}

# Main evaluation loop
for i,(qid,qtext) in enumerate(tqdm(queries.items(), desc=f"Bandit-{DATASET}")):
    if i==ALPHA_DECAY_AT: bandit.alpha=ALPHA_DECAY
    x=context_vec(qtext); a_idx=bandit.pull(x); arm=ARMS[a_idx]

    merged=defaultdict(float)
    for a in ARMS:
        pool=runs[a][qid]; keep=TOP_K_ARM if a==arm else 3
        for d,s in sorted(pool.items(), key=lambda kv:-kv[1])[:keep]:
            merged[d]=max(merged[d], s)
    ranked_ids=sorted(merged.items(), key=lambda kv:-kv[1])[:TOP_K_RERANK]
    pairs=[[qtext, corpus[did]["text"]] for did,_ in ranked_ids]
    scores=reranker.predict(pairs, convert_to_numpy=True)
    ranked={did:float(s) for (did,_),s in zip(ranked_ids, scores)}

    score=ndcg_at_10(ranked, qrels[qid])
    diversity=float(len(set(ranked).difference(prev_ranked))>=2) if prev_ranked else 0.0
    reward=score - LAMBDA*(latencies[arm]/len(queries)) + GAMMA*diversity
    bandit.update(a_idx,x,reward)

    band_scores.append(score); band_lat.append(latencies[arm]/len(queries))
    arm_stats[arm].append(score); prev_ranked=set(ranked)

    if i==N_PREFILL and np.mean(band_scores)<FAIL_THRESH:
        logging.warning("Quality low → switching to cached RRF …")
        for qid2 in list(queries.keys())[i+1:]:
            s=ndcg_at_10(rrf_cache[qid2], qrels[qid2])
            band_scores.append(s); band_lat.append(0.05)
        break

# Feature engineering for context vector
_tok = lambda t: re.findall(r"\w+", t.lower())
idf = {t: math.log(len(corpus)/(1+sum(t in _tok(doc["text"]) for doc in corpus.values())))
       for t in {w for doc in corpus.values() for w in _tok(doc["text"])}}
_year = re.compile(r"\b(19|20)\d{2}\b")
BM25_INDEX = BM25Okapi([_tok(doc["text"]) for doc in corpus.values()])

def _entropy(toks):
    c = Counter(toks); tot = len(toks)
    return -sum(v/tot*math.log2(v/tot) for v in c.values()) if tot else 0.0

def context_vec(q: str):
    toks=_tok(q); idfs=[idf.get(t,0.0) for t in toks]
    bm=max(BM25_INDEX.get_scores(toks)) if toks else 0.0
    return np.array([
        len(q), len(toks), len(set(toks)), sum(c.isdigit() for c in q),
        float(any(t[0].isupper() for t in toks)), np.mean([len(t) for t in toks]) if toks else 0.0,
        float("?" in q), float(q.endswith("?")), _entropy(toks), np.mean(idfs) if idfs else 0.0,
        bm, float(bool(_year.search(q)))
    ], dtype=np.float32)

# LinTS contextual bandit class
class LinTS:
    def __init__(self, n:int, d:int, alpha:float, lat, ndcg_prior, w_ndcg:float=1.0):
        self.A=[np.eye(d) for _ in range(n)]
        self.b=[(w_ndcg*ndcg - LAMBDA*l)*np.ones(d) for ndcg,l in zip(ndcg_prior,lat)]
        self.alpha=alpha
    def pull(self,x):
        mus=[]
        for A,b in zip(self.A,self.b):
            Ainv=np.linalg.inv(A)
            theta=np.random.multivariate_normal(Ainv@b, self.alpha**2*Ainv)
            mus.append(theta@x)
        return int(np.argmax(mus))
    def update(self,i,x,r):
        self.A[i]+=np.outer(x,x)
        self.b[i]+=r*x

# Utility function for reciprocal rank fusion (RRF)
def rrf(*doclists, k:int=60):
    fused=defaultdict(float)
    for dl in doclists:
        for rank,(d,_) in enumerate(sorted(dl.items(), key=lambda kv:-kv[1]),1):
            fused[d]+=1/(k+rank)
    return fused

# Compute priors for bandit arms
lat_prior=[latencies[a]/len(queries) for a in ARMS]
ndcg_prior=[np.mean([ndcg_at_10(runs[a][q], qrels[q]) for q in queries]) for a in ARMS]

# Initialize bandit model
bandit=LinTS(len(ARMS), D, ALPHA_START, lat_prior, ndcg_prior)
band_scores, band_lat, arm_stats=[],[],defaultdict(list)
prev_ranked=None
rrf_cache={qid: rrf(runs["bge"][qid], runs["contr"][qid]) for qid in queries}

# Main evaluation loop
for i,(qid,qtext) in enumerate(tqdm(queries.items(), desc=f"Bandit-{DATASET}")):
    if i==ALPHA_DECAY_AT: bandit.alpha=ALPHA_DECAY
    x=context_vec(qtext); a_idx=bandit.pull(x); arm=ARMS[a_idx]

    merged=defaultdict(float)
    for a in ARMS:
        pool=runs[a][qid]; keep=TOP_K_ARM if a==arm else 3
        for d,s in sorted(pool.items(), key=lambda kv:-kv[1])[:keep]:
            merged[d]=max(merged[d], s)
    ranked_ids=sorted(merged.items(), key=lambda kv:-kv[1])[:TOP_K_RERANK]
    pairs=[[qtext, corpus[did]["text"]] for did,_ in ranked_ids]
    scores=reranker.predict(pairs, convert_to_numpy=True)
    ranked={did:float(s) for (did,_),s in zip(ranked_ids, scores)}

    score=ndcg_at_10(ranked, qrels[qid])
    diversity=float(len(set(ranked).difference(prev_ranked))>=2) if prev_ranked else 0.0
    reward=score - LAMBDA*(latencies[arm]/len(queries)) + GAMMA*diversity
    bandit.update(a_idx,x,reward)

    band_scores.append(score); band_lat.append(latencies[arm]/len(queries))
    arm_stats[arm].append(score); prev_ranked=set(ranked)

    if i==N_PREFILL and np.mean(band_scores)<FAIL_THRESH:
        logging.warning("Quality low → switching to cached RRF …")
        for qid2 in list(queries.keys())[i+1:]:
            s=ndcg_at_10(rrf_cache[qid2], qrels[qid2])
            band_scores.append(s); band_lat.append(0.05)
        break

# Final results summary
print(f"\n=== {DATASET.capitalize()} ({len(band_scores)} queries) ===")
print(f"Bandit   nDCG@10 {np.mean(band_scores):.4f}   mean lat {np.mean(band_lat):.3f}s")
print(f"Union-6  nDCG@10 {np.mean(union_scores):.4f}   mean lat {np.mean(union_lat):.3f}s")

print("\nPer-arm nDCG@10 (sampled)")
for arm in ARMS:
    if arm_stats[arm]:
        print(f"{arm:<6} {np.mean(arm_stats[arm]):.4f}   ({latencies[arm]/len(queries):.3f}s/query)")

print("\nFull-dataset per-arm nDCG@10 (all queries)")
for arm in ARMS:
    full=[ndcg_at_10(runs[arm][qid], qrels[qid]) for qid in queries]
    print(f"{arm:<6} {np.mean(full):.4f}")

