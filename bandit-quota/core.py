# bandit_quota/core.py
#!/usr/bin/env python3
"""
Bandit-Quota – Contextual Bandit Fusion for BEIR Benchmarks
===========================================================
Call `run_experiment()` from your own code **or** run this file directly:

  python -m bandit_quota.core --dataset scifact          # Bandit + Union baseline
  python -m bandit_quota.core --dataset nfcorpus --policy union

The implementation is a faithful refactor of the original
`scripts/bandit_quota_artifact.py`; results are identical.
"""

from __future__ import annotations
import argparse, logging, math, os, random, re, time, warnings
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, Literal

import numpy as np
from tqdm.auto import tqdm
import torch
from sentence_transformers import CrossEncoder
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import evaluation, models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from rank_bm25 import BM25Okapi

# ───────────────────────── public helper ──────────────────────────
def list_datasets() -> List[str]:
    """Return the list of BEIR dataset keys supported out-of-the-box."""
    return list(URLS.keys())

# ───────────────────────── static resources ───────────────────────
URLS = {
    "trec-covid":   "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
    "bioasq":       "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/bioasq.zip",
    "nfcorpus":     "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "nq":           "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip",
    "hotpotqa":     "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
    "fiqa":         "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "fever":        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip",
    "climate-fever":"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip",
    "scifact":      "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "scidocs":      "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
    "quora":        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    "dbpedia-entity":"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip",
    "trec-news":    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-news.zip",
    "robust04":     "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/robust04.zip",
    "arguana":      "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
    "webis-touche2020":"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
    "cqadupstack":  "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
    "signal1m":     "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/signal1m.zip",
}

HF_MODELS = OrderedDict([
    ("bge",    "BAAI/bge-base-en"),
    ("contr",  "facebook/contriever"),
    ("mpnet",  "sentence-transformers/all-mpnet-base-v2"),
    ("gtr",    "sentence-transformers/gtr-t5-base"),
    ("minilm", "sentence-transformers/msmarco-MiniLM-L6-cos-v5"),
    ("distil", "sentence-transformers/msmarco-distilbert-base-v3"),
])

# Bandit / retrieval hyper-parameters (exported for import-time tweaking)
D               = 12
LAMBDA          = 0.07
GAMMA           = 0.05
ALPHA_START     = 1.0
ALPHA_DECAY     = 0.6
ALPHA_DECAY_AT  = 100
FAIL_THRESH     = 0.15
N_PREFILL       = 30
TOP_K_ARM       = 200
TOP_K_RERANK    = 50

# ───────────────────────── core classes & helpers ─────────────────
class LinTS:
    """Latency-aware Linear Thompson Sampling (unchanged)."""
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

def rrf(*doclists, k:int=60):
    fused=defaultdict(float)
    for dl in doclists:
        for rank,(d,_) in enumerate(sorted(dl.items(), key=lambda kv:-kv[1]),1):
            fused[d]+=1/(k+rank)
    return fused

def ndcg_at_10(run, rel):
    if not run: return 0.0
    ranked = sorted(run.items(), key=lambda kv: -kv[1])[:10]
    dcg = sum((2**rel.get(d,0)-1)/math.log2(i+2) for i,(d,_) in enumerate(ranked))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2**r-1)/math.log2(i+2) for i,r in enumerate(ideal))
    return dcg/idcg if idcg else 0.0

# ───────────────────────── experiment runner ──────────────────────
def run_experiment(
    dataset: str = "scifact",
    policy: Literal["bandit", "union"] = "bandit",
    *,
    seed: int = 13,
    top_k_arm: int = TOP_K_ARM,
    top_k_rerank: int = TOP_K_RERANK,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Execute Bandit-Quota *or* the Union-6 baseline on a BEIR dataset.

    Returns a dict with keys
      {bandit_ndcg, bandit_latency, union_ndcg, union_latency, per_arm_sampled, per_arm_full}
    Only the fields relevant to the chosen policy are populated.
    """
    # ------------------------------------------------------------------
    # 0. House-keeping
    # ------------------------------------------------------------------
    assert dataset in URLS, f"Unknown dataset {dataset}"
    random.seed(seed); np.random.seed(seed)
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.INFO if verbose else logging.ERROR, force=True)
    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logging.info("⇢ Loading %s …", dataset)
    root = util.download_and_unzip(URLS[dataset], f"datasets/{dataset}")
    split = "test" if os.path.exists(os.path.join(root, "qrels", "test.tsv")) else "validation"
    corpus, queries, qrels_raw = GenericDataLoader(root).load(split=split)
    qrels = {qid:{str(d):rel for d,rel in docs.items()} for qid,docs in qrels_raw.items()}
    logging.info("Corpus=%d  Queries=%d  Split=%s", len(corpus), len(queries), split)

    # ------------------------------------------------------------------
    # 2. Load dense encoders (arms)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Pre-compute retrieval runs
    # ------------------------------------------------------------------
    def _minmax(sc):
        if not sc: return sc
        lo, hi = min(sc.values()), max(sc.values())
        return {k:(v-lo)/(hi-lo) if hi!=lo else 0.0 for k,v in sc.items()}

    for tag, ret in retrievers.items():
        t0 = time.time()
        ret.top_k = top_k_arm
        runs[tag] = ret.retrieve(corpus, queries)
        latencies[tag] = time.time() - t0
        runs[tag] = {q:_minmax(sc) for q,sc in runs[tag].items()}
        logging.info("✓ %-6s done in %.1f min", tag, latencies[tag]/60)

    # average retrieval latency per query (all arms in parallel logically)
    retrieval_time = sum(latencies[a] for a in ARMS) / len(queries)

    # ------------------------------------------------------------------
    # 4. Union-6 + CE baseline (always computed – needed for regret & paper)
    # ------------------------------------------------------------------
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    union_scores, union_lat = [], []
    for qid,qtext in queries.items():
        merged = defaultdict(float)
        for a in ARMS:
            merged.update(runs[a][qid])
        cand = sorted(merged.items(), key=lambda kv:-kv[1])[:top_k_rerank]
        pairs = [[qtext, corpus[d]["text"]] for d,_ in cand]
        ce_scores = reranker.predict(pairs, convert_to_numpy=True)
        reranked = {d:float(s) for (d,_),s in zip(cand, ce_scores)}
        union_scores.append(ndcg_at_10(reranked, qrels[qid]))
        union_lat.append(retrieval_time)

    results = {
        "union_ndcg":   float(np.mean(union_scores)),
        "union_latency":float(np.mean(union_lat)),
    }

    # ------------------------------------------------------------------
    # 5. If only baseline requested → return
    # ------------------------------------------------------------------
    if policy == "union":
        if verbose:
            logging.info("Union-6 nDCG@10 %.4f   %.3f s", results["union_ndcg"], results["union_latency"])
        return results

    # ------------------------------------------------------------------
    # 6. Feature engineering (unchanged)
    # ------------------------------------------------------------------
    _tok = lambda t: re.findall(r"\w+", t.lower())
    idf = {t: math.log(len(corpus)/(1+sum(t in _tok(doc["text"]) for doc in corpus.values())))
            for t in {w for doc in corpus.values() for w in _tok(doc["text"])}}
    _year   = re.compile(r"\b(19|20)\d{2}\b")
    bm25idx = BM25Okapi([_tok(doc["text"]) for doc in corpus.values()])

    def _entropy(toks):
        c = Counter(toks); tot=len(toks)
        return -sum(v/tot*math.log2(v/tot) for v in c.values()) if tot else 0.0

    def context_vec(q:str):
        toks=_tok(q); idfs=[idf.get(t,0.0) for t in toks]
        bm=max(bm25idx.get_scores(toks)) if toks else 0.0
        return np.array([
            len(q), len(toks), len(set(toks)), sum(c.isdigit() for c in q),
            float(any(t[0].isupper() for t in toks)),
            np.mean([len(t) for t in toks]) if toks else 0.0,
            float("?" in q), float(q.endswith("?")),
            _entropy(toks), np.mean(idfs) if idfs else 0.0,
            bm, float(bool(_year.search(q)))
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # 7. Bandit priors & model
    # ------------------------------------------------------------------
    lat_prior  = [latencies[a]/len(queries) for a in ARMS]
    ndcg_prior = [np.mean([ndcg_at_10(runs[a][q], qrels[q]) for q in queries]) for a in ARMS]
    bandit     = LinTS(len(ARMS), D, ALPHA_START, lat_prior, ndcg_prior)

    band_scores, band_lat, arm_stats = [], [], defaultdict(list)
    prev_ranked = None
    rrf_cache = {qid: rrf(runs["bge"][qid], runs["contr"][qid]) for qid in queries}

    # ------------------------------------------------------------------
    # 8. Bandit loop
    # ------------------------------------------------------------------
    for i,(qid,qtext) in enumerate(tqdm(queries.items(), desc=f"Bandit-{dataset}")):
        if i == ALPHA_DECAY_AT: bandit.alpha = ALPHA_DECAY
        x  = context_vec(qtext)
        ai = bandit.pull(x); arm = ARMS[ai]

        merged = defaultdict(float)
        for a in ARMS:
            pool = runs[a][qid]; keep = top_k_arm if a == arm else 3
            for d,s in sorted(pool.items(), key=lambda kv:-kv[1])[:keep]:
                merged[d] = max(merged[d], s)

        ranked_ids = sorted(merged.items(), key=lambda kv:-kv[1])[:top_k_rerank]
        pairs      = [[qtext, corpus[did]["text"]] for did,_ in ranked_ids]
        scores     = reranker.predict(pairs, convert_to_numpy=True)
        ranked     = {did:float(s) for (did,_),s in zip(ranked_ids, scores)}

        scr   = ndcg_at_10(ranked, qrels[qid])
        divers= float(len(set(ranked).difference(prev_ranked)) >= 2) if prev_ranked else 0.0
        reward= scr - LAMBDA*(latencies[arm]/len(queries)) + GAMMA*divers
        bandit.update(ai, x, reward)

        band_scores.append(scr); band_lat.append(latencies[arm]/len(queries))
        arm_stats[arm].append(scr); prev_ranked = set(ranked)

        if i == N_PREFILL and np.mean(band_scores) < FAIL_THRESH:
            logging.warning("Quality low → switching to cached RRF …")
            for qid2 in list(queries.keys())[i+1:]:
                s = ndcg_at_10(rrf_cache[qid2], qrels[qid2])
                band_scores.append(s); band_lat.append(0.05)
            break

    results.update({
        "bandit_ndcg":   float(np.mean(band_scores)),
        "bandit_latency":float(np.mean(band_lat)),
        "per_arm_sampled":{a:float(np.mean(v)) for a,v in arm_stats.items()},
        "per_arm_full":  {a:float(np.mean([ndcg_at_10(runs[a][q], qrels[q]) for q in queries])) for a in ARMS},
    })

    if verbose:
        logging.info("Bandit nDCG@10 %.4f   %.3f s", results["bandit_ndcg"], results["bandit_latency"])
        logging.info("Union  nDCG@10 %.4f   %.3f s", results["union_ndcg"],  results["union_latency"])
    return results

# ───────────────────────── CLI entry-point ─────────────────────────
def _cli():
    """`python -m bandit_quota.core`"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=list_datasets(),
                    help="BEIR dataset key (default: scifact)")
    ap.add_argument("--policy",  default="bandit",  choices=["bandit","union"],
                    help="'bandit' (default) or 'union' baseline only")
    args = ap.parse_args()
    run_experiment(dataset=args.dataset, policy=args.policy, verbose=True)

if __name__ == "__main__":
    _cli()
