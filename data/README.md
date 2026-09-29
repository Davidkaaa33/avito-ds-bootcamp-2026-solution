# Данные

Исходные данные задания и тяжёлые сгенерированные артефакты специально не хранятся в Git.

## Исходные файлы задания

В эту папку нужно положить:

```text
train.parquet
benchmark_queries.parquet
benchmark_items.parquet
```

## Генерируемые артефакты

Скрипт `build_train_assets.py` создаёт:

```text
train_bge_embeddings.npy
train_bge_item_ids.npy
train_bm25_index/
```

Скрипт `microcat_classifier.py` создаёт:

```text
microcat_vectorizer.joblib
microcat_classifier.joblib
```

Скрипт `build_benchmark_assets.py` создаёт:

```text
benchmark_bge_embeddings.npy
benchmark_bge_item_ids.npy
benchmark_bm25_index/
```

Порядок объявлений при подготовке артефактов сохраняется и совпадает с порядком, который использует финальный пайплайн поиска.
