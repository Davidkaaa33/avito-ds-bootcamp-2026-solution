from pathlib import Path
import bm25s
import numpy as np
import pandas as pd
import Stemmer
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent / "data"
ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
BGE_EMBEDDINGS_PATH = DATA_DIR / "benchmark_bge_embeddings.npy"
BGE_ITEM_IDS_PATH = DATA_DIR / "benchmark_bge_item_ids.npy"
BM25_INDEX_PATH = DATA_DIR / "benchmark_bm25_index"
ITEM_COLUMNS = ["item_title_raw", "item_infm_params_text", "item_description_raw", "item_id"]

def select_device():
    """Выбираем CUDA, потом MPS, в крайнем случае считаем на CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_items():
    """Берём уникальные benchmark items в том же порядке, который потом использует solution.py."""
    items = pd.read_parquet(ITEMS_PATH, columns=ITEM_COLUMNS)
    return items.drop_duplicates("item_id").reset_index(drop=True)

def build_texts(items):
    """Текст собираем точно так же, как для train, иначе train/benchmark retrieval будет несогласованным."""
    titles = items["item_title_raw"].fillna("").astype(str)
    params = items["item_infm_params_text"].fillna("").astype(str)
    descriptions = items["item_description_raw"].fillna("").astype(str).str.slice(0, 600)
    bge_texts = (titles + ". " + params + ". " + descriptions).tolist()
    # Для benchmark использую ровно ту же схему текста, что и для train.
    # Title также дублирую, чтобы lexical weighting не отличался между двумя корпусами.
    bm25_texts = (titles + " " + titles + " " + params + " " + descriptions).tolist()
    return bge_texts, bm25_texts

def ensure_bge_assets(items, texts):
    """Проверяем готовые embeddings или считаем их, если файлов ещё нет."""
    item_ids = items["item_id"].to_numpy()
    if BGE_EMBEDDINGS_PATH.exists() and BGE_ITEM_IDS_PATH.exists():
        embeddings = np.load(BGE_EMBEDDINGS_PATH, mmap_mode="r")
        saved_ids = np.load(BGE_ITEM_IDS_PATH, allow_pickle=True)
        if embeddings.shape[0] != len(items) or not np.array_equal(saved_ids, item_ids):
            raise RuntimeError("Existing benchmark BGE assets do not match benchmark item order.")
        print("Benchmark BGE assets already exist and are consistent.")
        return
    if BGE_EMBEDDINGS_PATH.exists() != BGE_ITEM_IDS_PATH.exists():
        raise RuntimeError("Only one benchmark BGE artifact exists. Remove it and rerun.")
    device = select_device()
    print("Building benchmark BGE embeddings on", device)
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    model.max_seq_length = 128
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    np.save(BGE_EMBEDDINGS_PATH, embeddings)
    np.save(BGE_ITEM_IDS_PATH, item_ids)
    print("Saved:", BGE_EMBEDDINGS_PATH)
    print("Saved:", BGE_ITEM_IDS_PATH)

def ensure_bm25_index(texts):
    """Строим BM25 индекс для benchmark corpus, повторно его считать уже не нужно."""
    if BM25_INDEX_PATH.exists():
        print("Benchmark BM25 index already exists; skipping.")
        return
    print("Building benchmark BM25 index...")
    stemmer = Stemmer.Stemmer("russian")
    corpus_tokens = bm25s.tokenize(texts, stopwords=None, stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(BM25_INDEX_PATH))
    print("Saved:", BM25_INDEX_PATH)



# Здесь воспроизводится подготовка benchmark-части: сначала фиксируется порядок items,
# затем для этого же порядка строятся BGE embeddings и BM25 index.
def main():
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {ITEMS_PATH}")
    DATA_DIR.mkdir(exist_ok=True)
    items = load_items()
    bge_texts, bm25_texts = build_texts(items)
    print("Unique benchmark items:", len(items))
    ensure_bge_assets(items, bge_texts)
    ensure_bm25_index(bm25_texts)
    print("Benchmark assets are ready.")
if __name__ == "__main__":
    main()
