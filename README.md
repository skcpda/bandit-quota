# Bandit‑Quota (6‑arm) — BEIR SciFact

A lightweight **contextual‑bandit retrieval** demo that combines six off‑the‑shelf dense encoders with a latency‑aware Thompson‑sampling policy.

![optional alt text](bandit-quota/bandit-quota/Bandit.png)


The pipeline reproduces the headline results reported in our CIKM 2025 resource‑track submission:

```
Bandit    nDCG@10 ≈ 0.704   mean latency ≈ 0.91 s/query
Union‑6   nDCG@10 ≈ 0.491   mean latency ≈ 6.97 s/query
```

Everything lives in a single, self‑contained script — **`scripts/bandit_quota_artifact.py`** — that you can run on any CPU‑only machine with ≥16 GB RAM.

---

## Requirements

* Python 3.9 – 3.12
* `pip install -r requirements.txt` (≈ 900 MB once all HF models are cached)
* **No GPU needed** — the reranker and encoders run comfortably on a modern laptop.

> **Tip:** first run with `TRANSFORMERS_OFFLINE=1` if you have already cached the models elsewhere.

---

## Quick‑start

```bash
# 1) clone and enter
$ git clone https://github.com/your‑name/bandit‑quota
$ cd bandit‑quota

# 2) (optional) create virtual‑env
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# 3) run the artifact script
$ python scripts/bandit_quota_artifact.py
```

The script automatically downloads the **BEIR SciFact** test split (\~9 MB) on first launch, produces per‑arm baselines, the naïve union run, and the Bandit‑Quota scores.

Expected terminal tail:

```
=== SciFact test (300 queries) ===
Bandit    nDCG@10 0.7043   mean lat 0.907s
Union‑6   nDCG@10 0.4908   mean lat 6.970s
...
```

---

## Repository layout (minimal)

```
├── scripts/
│   └── bandit_quota_artifact.py   ← the only executable
├── LICENSE                        ← MIT
├── README.md                      ← you are here
└── requirements.txt               ← pinned deps (BEIR v2, Sentence‑Transformers ≥ 2.5)
```

---

## Citation

If you build on this work, please cite the resource paper: (To be updated soon)

---

## License

Released under the MIT License — see the `LICENSE` file for full text.
