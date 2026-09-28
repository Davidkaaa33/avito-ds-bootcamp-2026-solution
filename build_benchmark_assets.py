from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer
import torch

from sentence_transformers import SentenceTransformer

def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = select_device()

DATA_DIR = Path(__file__).resolve().parent / "data"

ITEM_COLUMNS = [
    "item_title_raw",
    "item_infm_params_text",
    "item_description_raw",
    "item_id",
]

BGE_EMBEDDINGS_PATH = DATA_DIR / "benchmark_bge_embeddings.npy"

BGE_ITEM_IDS_PATH = DATA_DIR / "benchmark_bge_item_ids.npy"

BM25_INDEX_PATH = DATA_DIR / "benchmark_bm25_index"

print("Загружаем benchmark_items...")

items = pd.read_parquet(
    DATA_DIR / "benchmark_items.parquet",
    columns=ITEM_COLUMNS,
)

items = (
    items
    .drop_duplicates("item_id")
    .reset_index(drop=True)
)

print( "Benchmark items:",len(items))

titles = (
    items["item_title_raw"]
    .fillna("")
    .astype(str)
)

params = (
    items["item_infm_params_text"]
    .fillna("")
    .astype(str)
)

descriptions = (
    items["item_description_raw"]
    .fillna("")
    .astype(str)
    .str.slice(0, 600)
)

bge_texts = (
    titles
    + ". "
    + params
    + ". "
    + descriptions
).tolist()

if (
    BGE_EMBEDDINGS_PATH.exists()
    and BGE_ITEM_IDS_PATH.exists()
):

    print("\nBenchmark BGE embeddings уже существуют.")

    embeddings = np.load(
        BGE_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    saved_ids = np.load(
        BGE_ITEM_IDS_PATH,
        allow_pickle=True,
    )

    assert embeddings.shape[0] == len(items)

    assert np.array_equal(
        saved_ids,
        items["item_id"].to_numpy(),
    )

    print("Пропускаем BGE.")

else:

    print("\nЗагружаем BGE-M3...")

    model = SentenceTransformer(
        "BAAI/bge-m3",
        device=DEVICE,
    )

    model.max_seq_length = 128

    print("Считаем embeddings для benchmark items...")

    embeddings = model.encode(
        bge_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(
        BGE_EMBEDDINGS_PATH,
        embeddings,
    )

    np.save(
        BGE_ITEM_IDS_PATH,
        items["item_id"].to_numpy(),
    )

    print("BGE embeddings сохранены:")

    print(BGE_EMBEDDINGS_PATH)

    print("Shape:", embeddings.shape)

bm25_texts = (
    titles
    + " "
    + titles
    + " "
    + params
    + " "
    + descriptions
).tolist()

if BM25_INDEX_PATH.exists():

    print("\nBM25 benchmark index уже существует.")

    print("Пропускаем BM25.")

else:

    print("\nТокенизируем benchmark corpus...")

    stemmer = Stemmer.Stemmer(
        "russian"
    )

    corpus_tokens = bm25s.tokenize(
        bm25_texts,
        stopwords=None,
        stemmer=stemmer,
    )

    print("Строим BM25 index...")

    retriever = bm25s.BM25()

    retriever.index(
        corpus_tokens
    )

    print("Сохраняем BM25 index...")

    retriever.save(
        str(
            BM25_INDEX_PATH
        )
    )

    print("BM25 index сохранён:")

    print(BM25_INDEX_PATH)

print( "\n" + "=" * 70)

print("BENCHMARK ASSETS READY")

print( "=" * 70)

embeddings = np.load(
    BGE_EMBEDDINGS_PATH,
    mmap_mode="r",
)

saved_ids = np.load(
    BGE_ITEM_IDS_PATH,
    allow_pickle=True,
)

print( "Items:",len(items))

print("BGE shape:", embeddings.shape)

print( "Item IDs:",len(saved_ids))

print("BM25 index:", BM25_INDEX_PATH)

assert (
    embeddings.shape[0]
    == len(items)
)

assert (
    len(saved_ids)
    == len(items)
)

assert np.array_equal(
    saved_ids,
    items["item_id"].to_numpy(),
)

print()
print("Все проверки пройдены.")
