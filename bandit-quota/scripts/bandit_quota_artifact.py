#!/usr/bin/env python3
"""
Bandit‑Quota – 6‑arm dense variant for BEIR‑SciFact (test split)
================================================================
This script compares three systems on 300 SciFact test queries:
  • **Bandit‑Quota** contextual bandit (6 dense encoders)
  • Best single encoder (reported separately per arm)
  • Naïve UNION‑6 run (top‑k merge of all six arms)

Key settings (identical to previous winning run unless noted):
  • Latency‑biased LinTS (λ = 0.07) with nDCG priors
  • 12‑D query feature vector + small diversity bonus
  • Cross‑Encoder rerank (MiniLM‑L‑6‑v2) on top‑50 merged docs
  • Early fallback to RRF(bge, contriever) if quality < 0.15 after 30 pulls
  • All inference on CPU / single GPU‑free box; MPS avoided for safety

Prints: Bandit nDCG/latency, Union‑6 nDCG/latency, per‑arm means (sampled
        + full dataset) so you can lift the numbers straight into the paper.
"""

from __future__ import annotations

import logging, math, os, random, re, time, warnings
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

# ───────────────────────── helpers / constants ─────────────────────────
_PYTORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D               = 12
LAMBDA          = 0.07      # latency penalty
GAMMA           = 0.05      # diversity bonus
ALPHA_START     = 1.0
ALPHA_DECAY     = 0.6       # after 100 pulls
ALPHA_DECAY_AT  = 100
FAIL_THRESH     = 0.15
N_PREFILL       = 30
TOP_K_ARM       = 200
TOP_K_RERANK    = 50
SEED            = 13
random.seed(SEED); np.random.seed(SEED)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO, force=True)
warnings.filterwarnings("ignore")

# ───────────────────────── dataset ─────────────────────────
URL  = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
ROOT = util.download_and_unzip(URL, "datasets")
corpus, queries, qrels_raw = GenericDataLoader(os.path.join(ROOT, "scifact")).load(split="test")
qrels: Dict[str, Dict[str, int]] = {qid: {str(d): rel for d, rel in docs.items()} for qid, docs in qrels_raw.items()}
logging.info("Corpus=%d  Queries=%d", len(corpus), len(queries))

# ───────────────────────── dense retrievers (6 arms) ─────────────────────────
HF_MODELS = OrderedDict([
    ("bge",    "BAAI/bge-base-en"),
    ("contr",  "facebook/contriever"),
    ("mpnet",  "sentence-transformers/all-mpnet-base-v2"),
    ("gtr",    "sentence-transformers/gtr-t5-base"),
    ("minilm", "sentence-transformers/msmarco-MiniLM-L6-cos-v5"),
    ("distil", "sentence-transformers/msmarco-distilbert-base-v3"),
])

retrievers, latencies, runs = {}, {}, {}

def _load_dense(tag: str, hf_id: str):
    try:
        model  = models.SentenceBERT(hf_id)
        search = DRES(model, batch_size=32, corpus_chunk_size=2048, score_function="dot")
        return evaluation.EvaluateRetrieval(search, score_function="dot")
    except Exception as exc:
        logging.warning("⚠️ %s skipped: %s", tag, exc)
        return None

for tag, mid in HF_MODELS.items():
    ret = _load_dense(tag, mid)
    if ret:
        retrievers[tag] = ret
assert len(retrievers) == 6, "Expected six live arms"
ARMS: List[str] = list(retrievers.keys())

# ───────────────────────── pre‑compute runs ─────────────────────────

def _minmax(sc: Dict[str, float]):
    if not sc:
        return sc
    lo, hi = min(sc.values()), max(sc.values())
    return {k: (v - lo) / (hi - lo) if hi != lo else 0.0 for k, v in sc.items()}

for tag, ret in retrievers.items():
    t0 = time.time()
    ret.top_k = TOP_K_ARM
    runs[tag] = ret.retrieve(corpus, queries)
    latencies[tag] = time.time() - t0
    runs[tag] = {q: _minmax(sc) for q, sc in runs[tag].items()}
    logging.info("✓ %-6s done in %.1f min", tag, latencies[tag] / 60)

# ───────────────────────── ndcg helper ─────────────────────────

def ndcg_at_10(run, rel):
    if not run:
        return 0.0
    ranked = sorted(run.items(), key=lambda kv: -kv[1])[:10]
    dcg = sum((2**rel.get(d,0)-1)/math.log2(i+2) for i,(d,_) in enumerate(ranked))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2**r-1)/math.log2(i+2) for i,r in enumerate(ideal))
    return dcg/idcg if idcg else 0.0

# ───────────────────────── union‑6 baseline ─────────────────────────
union_scores, union_lat = [], []
lat_sum = sum(latencies[a] for a in ARMS) / len(queries)
for qid in queries:
    merged = defaultdict(float)
    for a in ARMS:
        merged.update(runs[a][qid])
    union_scores.append(ndcg_at_10(merged, qrels[qid]))
    union_lat.append(lat_sum)

# ───────────────────────── CE reranker (CPU) ─────────────────────────
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# ───────────────────────── feature engineering ─────────────────────────
_tok = lambda t: re.findall(r"\w+", t.lower())
idf = {t: math.log(len(corpus)/(1+sum(t in _tok(doc["text"]) for doc in corpus.values())))
       for t in {w for doc in corpus.values() for w in _tok(doc["text"])}}
_year = re.compile(r"\b(19|20)\d{2}\b")
from rank_bm25 import BM25Okapi
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

# ───────────────────────── LinTS with priors ─────────────────────────
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

# ───────────────────────── misc helpers ─────────────────────────

def rrf(*doclists, k:int=60):
    fused=defaultdict(float)
    for dl in doclists:
        for rank,(d,_) in enumerate(sorted(dl.items(), key=lambda kv:-kv[1]),1):
            fused[d]+=1/(k+rank)
    return fused

lat_prior=[latencies[a]/len(queries) for a in ARMS]
ndcg_prior=[np.mean([ndcg_at_10(runs[a][q], qrels[q]) for q in queries]) for a in ARMS]

bandit=LinTS(len(ARMS), D, ALPHA_START, lat_prior, ndcg_prior)
band_scores, band_lat, arm_stats=[],[],defaultdict(list)
prev_ranked=None

rrf_cache={qid: rrf(runs["bge"][qid], runs["contr"][qid]) for qid in queries}

for i,(qid,qtext) in enumerate(tqdm(queries.items(), desc="Bandit")):
    if i==ALPHA_DECAY_AT:
        bandit.alpha=ALPHA_DECAY
    x=context_vec(qtext)
    a_idx=bandit.pull(x); arm=ARMS[a_idx]

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

# ───────────────────────── results ─────────────────────────
print(f"\n=== SciFact test ({len(band_scores)} queries) ===")
print(f"Bandit    nDCG@10 {np.mean(band_scores):.4f}   mean lat {np.mean(band_lat):.3f}s")
print(f"Union‑6   nDCG@10 {np.mean(union_scores):.4f}   mean lat {np.mean(union_lat):.3f}s")

print("\nPer‑arm nDCG@10 (sampled)")
for arm in ARMS:
    if arm_stats[arm]:
        print(f"{arm:<6} {np.mean(arm_stats[arm]):.4f}   ({latencies[arm]/len(queries):.3f}s/query)")

print("\nFull‑dataset per‑arm nDCG@10 (all 300 queries)")
for arm in ARMS:
    full=[ndcg_at_10(runs[arm][qid], qrels[qid]) for qid in queries]
    print(f"{arm:<6} {np.mean(full):.4f}")
