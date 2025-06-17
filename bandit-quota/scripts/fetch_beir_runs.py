#!/usr/bin/env python
"""
Downloads the SPLADE-distil run and dev-split qrels for MS MARCO
from the ir-datasets mirrors on Hugging Face and copies them into
data/… and beir-qrels/… so the rest of the repo can rely on them.
Requires: `pip install huggingface_hub`
"""

from huggingface_hub import hf_hub_download
import shutil, pathlib, os

runs_id   = "ir-datasets/BEIR-Runs"
qrels_id  = "ir-datasets/BEIR-Qrels"
run_file  = "msmarco/splade_distil_beir_run"   # 27 MB, 6-col TREC
qrels_file= "msmarco/dev/qrels"                # 14 kB, 4-col TREC

dest_run   = pathlib.Path("data/beir_runs/msmarco.splade.trec")
dest_qrels = pathlib.Path("beir-qrels/msmarco.dev.trec")
dest_run.parent.mkdir(parents=True, exist_ok=True)
dest_qrels.parent.mkdir(parents=True, exist_ok=True)

print("Downloading run …")
p_run   = hf_hub_download(repo_id=runs_id,  repo_type="dataset",
                          filename=run_file,  resume_download=True)
print("Downloading qrels …")
p_qrels = hf_hub_download(repo_id=qrels_id, repo_type="dataset",
                          filename=qrels_file, resume_download=True)

shutil.copy(p_run,   dest_run)
shutil.copy(p_qrels, dest_qrels)
print("✓ Saved", dest_run)
print("✓ Saved", dest_qrels)
