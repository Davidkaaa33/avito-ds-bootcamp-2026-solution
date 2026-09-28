from pathlib import Path
import bm25s
import numpy as np
import pandas as pd
import Stemmer
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "train.parquet"
BGE_EMBEDDINGS_PATH = DATA_DIR / "train_bge_embeddings.npy"
BGE_ITEM_IDS_PATH = DATA_DIR / "train_bge_item_ids.npy"
BM25_INDEX_PATH = DATA_DIR / "train_bm25_index"
ITEM_COLUMNS = ["item_title_raw", "item_infm_params_text", "item_description_raw", "item_id"]

def select_device():
    """Выбираем CUDA, потом MPS, а если ускорителя нет — обычный CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_items():
    """Берём уникальные train items и сохраняем тот же порядок, в котором будут лежать embeddings."""
    items = pd.read_parquet(TRAIN_PATH, columns=ITEM_COLUMNS)
    return items.drop_duplicates("item_id").reset_index(drop=True)

def build_texts(items):
    """Собираем два варианта текста: один для BGE, второй для BM25."""
    titles = items["item_title_raw"].fillna("").astype(str)
    params = items["item_infm_params_text"].fillna("").astype(str)
    descriptions = items["item_description_raw"].fillna("").astype(str).str.slice(0, 600)
    bge_texts = (titles + ". " + params + ". " + descriptions).tolist()
    # Title дублирую специально: так он получает немного больший вес в BM25, и это лучше работало на validation.
    bm25_texts = (titles + " " + titles + " " + params + " " + descriptions).tolist()
    return bge_texts, bm25_texts

def ensure_bge_assets(items, texts):
    """Если embeddings уже есть — проверяем их, иначе считаем заново."""
    item_ids = items["item_id"].to_numpy()
    if BGE_EMBEDDINGS_PATH.exists() and BGE_ITEM_IDS_PATH.exists():
        embeddings = np.load(BGE_EMBEDDINGS_PATH, mmap_mode="r")
        saved_ids = np.load(BGE_ITEM_IDS_PATH, allow_pickle=True)
        if embeddings.shape[0] != len(items) or not np.array_equal(saved_ids, item_ids):
            raise RuntimeError("Existing train BGE assets do not match train.parquet item order.")
        print("Train BGE assets already exist and are consistent.")
        return
    if BGE_EMBEDDINGS_PATH.exists() != BGE_ITEM_IDS_PATH.exists():
        raise RuntimeError("Only one train BGE artifact exists. Remove it and rerun.")
    device = select_device()
    print("Building train BGE embeddings on", device)
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    model.max_seq_length = 128
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    np.save(BGE_EMBEDDINGS_PATH, embeddings)
    np.save(BGE_ITEM_IDS_PATH, item_ids)
    print("Saved:", BGE_EMBEDDINGS_PATH)
    print("Saved:", BGE_ITEM_IDS_PATH)

def ensure_bm25_index(texts):
    """BM25 индекс строим один раз и дальше просто переиспользуем."""
    if BM25_INDEX_PATH.exists():
        print("Train BM25 index already exists; skipping.")
        return
    print("Building train BM25 index...")
    stemmer = Stemmer.Stemmer("russian")
    corpus_tokens = bm25s.tokenize(texts, stopwords=None, stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(BM25_INDEX_PATH))
    print("Saved:", BM25_INDEX_PATH)



# Здесь уже просто последовательная сборка всех train-артефактов.
def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {TRAIN_PATH}")
    DATA_DIR.mkdir(exist_ok=True)
    items = load_items()
    bge_texts, bm25_texts = build_texts(items)
    print("Unique train items:", len(items))
    ensure_bge_assets(items, bge_texts)
    ensure_bm25_index(bm25_texts)
    print("Train assets are ready.")
if __name__ == "__main__":
    main()
