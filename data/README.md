# Data and generated assets

The original competition parquet files and heavy retrieval artifacts are intentionally not stored in Git.

This directory acts as the input/output contract for the retrieval pipeline.

## Source files

Place these files in `data/` before running the project:

```text
train.parquet
benchmark_queries.parquet
benchmark_items.parquet
```

### `train.parquet`

Used for:

- query / item interaction history;
- query → microcategory supervision;
- geographic priors;
- repeated-query signals;
- historical item prototypes;
- offline validation experiments.

### `benchmark_queries.parquet`

Used as the benchmark query set. The final `answer.csv` preserves its query order exactly.

### `benchmark_items.parquet`

Defines the candidate item corpus for the benchmark. Every item ID returned by the final submission must exist in this file.

---

## Generated training assets

Run:

```bash
python build_train_assets.py
```

This creates:

```text
train_bge_embeddings.npy
train_bge_item_ids.npy
train_bm25_index/
```

The embedding file and item-ID file are positionally aligned. The builder checks existing artifacts before reusing them.

---

## Generated microcategory assets

Run:

```bash
python microcat_classifier.py
```

This creates:

```text
microcat_vectorizer.joblib
microcat_classifier.joblib
```

The model uses TF-IDF word/character n-grams with LinearSVC and a query-disjoint validation split.

---

## Generated benchmark assets

Run:

```bash
python build_benchmark_assets.py
```

This creates:

```text
benchmark_bge_embeddings.npy
benchmark_bge_item_ids.npy
benchmark_bm25_index/
```

The item ordering used while building these artifacts is the same ordering expected by `solution.py`.

---

## Why large artifacts are excluded

Embeddings and retrieval indexes are derived artifacts rather than source code. Keeping them out of Git:

- avoids repository bloat;
- keeps the code review surface small;
- makes the build process explicit;
- allows the same artifacts to be regenerated from the source parquet files.

The repository therefore stores the **builders and validation logic**, not large binary outputs.
