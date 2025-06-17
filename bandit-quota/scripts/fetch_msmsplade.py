from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.evaluation import EvaluateRetrieval
import os

#### Download and load dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
data_path = util.download_and_unzip(url, out_dir="datasets")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Load predefined retriever (SPLADE distil)
model = models.SentenceBERT("naver/splade-cocondenser-ensembledistil")
retriever = DenseRetrievalExactSearch(model, batch_size=64)

#### Run retrieval + evaluation
retrieval = EvaluateRetrieval(retriever)
results = retrieval.retrieve(corpus, queries)
ndcg, map_, recall, precision = retrieval.evaluate(qrels, results, [10])
print("nDCG@10:", ndcg["NDCG@10"])
