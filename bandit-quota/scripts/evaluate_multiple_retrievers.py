import os, time, logging
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

logging.basicConfig(level=logging.INFO)

# ---------------- 1. load SciFact test split ----------------
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
root = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(os.path.join(root, "scifact")).load(split="test")
qrels = {q: {d: int(r) for d, r in docs.items()} for q, docs in qrels.items()}

# ---------------- 2. define retrievers ----------------
retrievers = {
    "e5"      : "intfloat/e5-base-v2",
    "bge"     : "BAAI/bge-base-en",
    "contr"   : "facebook/contriever",
    "colbert" : "colbert-ir/colbertv2.0",
    "ance"    : "sentence-transformers/msmarco-roberta-base-ance-firstp",
}

# ---------------- 3. run and log ----------------
report = {}
print("\n=== SciFact test — nDCG@10 ===\nretriever   nDCG@10   time")

for name, hf_id in retrievers.items():
    t0 = time.time()
    model = models.SentenceBERT(hf_id)
    searcher = DRES(model, batch_size=32, corpus_chunk_size=2048, score_function="dot")
    ev      = EvaluateRetrieval(searcher, score_function="dot")
    results = ev.retrieve(corpus, queries)
    ndcg, *_ = ev.evaluate(qrels, results, k_values=[10])
    score    = ndcg["NDCG@10"]
    mins     = (time.time() - t0)/60
    print(f"{name:<10}  {score:7.4f}   {mins:4.1f} min")
    report[name] = score
