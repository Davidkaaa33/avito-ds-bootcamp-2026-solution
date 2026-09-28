# Data

The original task data and generated heavy artifacts are intentionally excluded from Git.

## Input files supplied by the task

Place these files in this directory:

```text
train.parquet
benchmark_queries.parquet
benchmark_items.parquet
```

## Generated artifacts

`build_train_assets.py` creates:

```text
train_bge_embeddings.npy
train_bge_item_ids.npy
train_bm25_index/
```

`microcat_classifier.py` creates:

```text
microcat_vectorizer.joblib
microcat_classifier.joblib
```

`build_benchmark_assets.py` creates:

```text
benchmark_bge_embeddings.npy
benchmark_bge_item_ids.npy
benchmark_bm25_index/
```

The builders preserve the item order used by the final retrieval pipeline.
