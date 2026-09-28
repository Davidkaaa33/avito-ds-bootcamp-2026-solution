# Avito Data Science Bootcamp 2026

Candidate-generation solution for the services-search task.

**Final leaderboard Recall@50: 0.831562**

## Task

For each search query, return 50 candidate service listings. The target metric is Recall@50, so the retrieval stage is optimized for broad coverage while keeping the final candidate set fixed at 50 items.

## Final approach

The final pipeline combines several complementary signals:

- BM25 lexical retrieval over title, parameters, and description;
- BAAI/bge-m3 semantic retrieval;
- exact-location retrieval;
- alternative-location retrieval based on historical location transitions;
- a geographic prior P(item_location | search_location);
- a query-to-microcategory classifier;
- joint location × microcategory retrieval;
- exact-query historical positives;
- history-prototype retrieval from embeddings of previously relevant items;
- weighted Reciprocal Rank Fusion plus small location, microcategory, and geo-prior bonuses.

The strongest additions were accepted only after they improved a separate confirmation slice.

## Validation

Most tuning used a query-disjoint split so that validation queries did not appear in the supervision part of the split. Different non-overlapping slices were used for tuning and independent confirmation.

Ideas that were tested but not kept in the final pipeline included generic cross-encoder reranking, a fine-tuned reranker, field-specific BM25, conditional geo × microcategory priors, and query-to-query transfer.

## Repository layout

- `solution.py` — final retrieval and fusion pipeline;
- `build_train_assets.py` — builds train BGE embeddings and BM25 index;
- `build_benchmark_assets.py` — builds benchmark BGE embeddings and BM25 index;
- `microcat_classifier.py` — trains the query-to-microcategory classifier;
- `check_submission.py` — validates submission format;
- `answer.csv` — exact submitted answer;
- `experiments/` — selected validation experiments;
- `data/README.md` — input and generated data layout.

## Reproduce from scratch

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the original task files into `data/`:

```text
data/train.parquet
data/benchmark_queries.parquet
data/benchmark_items.parquet
```

Build derived artifacts:

```bash
python build_train_assets.py
python microcat_classifier.py
python build_benchmark_assets.py
```

Generate and validate the submission:

```bash
python solution.py
python check_submission.py
```

The final output is `answer.csv`.

Large task datasets, embeddings, model artifacts, and BM25 indexes are intentionally excluded from Git.

## Final result

```text
Recall@50: 0.831562
```

SHA-256 of the submitted `answer.csv`:

```text
093072acc21b856f79a982cf67b1d7ffa9f57252387c1b99b4cdea7c66f05cd6
```
