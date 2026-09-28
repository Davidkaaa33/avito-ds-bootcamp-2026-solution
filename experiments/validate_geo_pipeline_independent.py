from pathlib import Path
from collections import defaultdict
import bm25s
import joblib
import numpy as np
import pandas as pd
import Stemmer
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
if ROOT.name == "data":
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.parquet"
BGE_EMB_PATH = DATA_DIR / "train_bge_embeddings.npy"
BGE_IDS_PATH = DATA_DIR / "train_bge_item_ids.npy"
BM25_DIR = DATA_DIR / "train_bm25_index"
MICROCAT_VECTORIZER_PATH = DATA_DIR / "microcat_vectorizer.joblib"
MICROCAT_CLASSIFIER_PATH = DATA_DIR / "microcat_classifier.joblib"
K_GLOBAL = 500
K_LOCAL_BGE = 500
K_LOCAL_BM25 = 500
K_MICROCAT = 500
K_BM25_WIDE = 10000
TOP_MICROCATS = 5
RRF_K = 60
BGE_GLOBAL_WEIGHT = 1.0
LOCAL_BGE_WEIGHT = 0.25
LOCAL_BM25_WEIGHT = 1.25
MICROCAT_SOURCE_WEIGHT = 0.25
LOCATION_BONUS = 0.020
MICROCAT_BONUS = 0.010
GEO_TOP_NS = [1, 3, 5]
GEO_WEIGHTS = [0.020, 0.035, 0.060, 0.080, 0.100, 0.120]
RANDOM_STATE = 42

def microcat_key(value):
    if pd.isna(value):
        return None
    try:
        x = float(value)
        if x.is_integer():
            return str(int(x))
    except (TypeError, ValueError):
        pass
    return str(value)

def add_rrf(scores, indices, weight):
    for rank, idx in enumerate(indices, start=1):
        idx = int(idx)
        scores[idx] = scores.get(idx, 0.0) + weight / (RRF_K + rank)

def top_k_indices(scores, indices, k):
    if len(indices) == 0:
        return np.array([], dtype=np.int64)
    k = min(k, len(indices))
    if k == len(indices):
        order = np.argsort(scores)[::-1]
        return indices[order]
    positions = np.argpartition(scores, -k)[-k:]
    order = np.argsort(scores[positions])[::-1]
    return indices[positions[order]]
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("Device:", DEVICE)
print("\nЗагружаем train...")
train = pd.read_parquet(TRAIN_PATH)
train["item_id"] = train["item_id"].astype(str)
print("Rows:", len(train))
print("\nСтроим query-disjoint split...")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_STATE)
train_idx, val_idx = next(splitter.split(train, groups=train["search_query"]))
supervision = train.iloc[train_idx].copy()
holdout = train.iloc[val_idx].copy()
query_overlap = set(supervision["search_query"]) & set(holdout["search_query"])
assert len(query_overlap) == 0
print("Supervision rows:", len(supervision))
print("Holdout rows:", len(holdout))
print("Query overlap:", len(query_overlap))
QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
val_groups = holdout.groupby(QUERY_COLUMNS, dropna=False)["item_id"].agg(set).reset_index(name="relevant_ids")
print("All holdout groups:", len(val_groups))
shuffled_val = val_groups.sample(frac=1, random_state=123).reset_index(drop=True)
if len(shuffled_val) < 4000:
    raise ValueError(f"Недостаточно holdout groups: " f"{len(shuffled_val)}")
sample = shuffled_val.iloc[1000:4000].copy().reset_index(drop=True)
print("Independent validation groups:", len(sample))
ITEM_COLUMNS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_location_id",
    "item_microcat_id",
]
items = train[ITEM_COLUMNS].drop_duplicates("item_id").reset_index(drop=True)
item_ids = items["item_id"].astype(str).to_numpy()
item_locations = items["item_location_id"].to_numpy()
item_microcats = np.asarray([microcat_key(x) for x in items["item_microcat_id"]], dtype=object)
print("Items:", len(items))
print("\nЗагружаем BGE embeddings...")
item_embeddings = np.load(BGE_EMB_PATH)
saved_item_ids = np.load(BGE_IDS_PATH, allow_pickle=True)
assert len(item_embeddings) == len(items)
assert np.array_equal(saved_item_ids.astype(str), item_ids), "Порядок item_id " "не совпадает с embeddings!"
print("Embeddings:", item_embeddings.shape)
print("\nСтроим location index...")
location_to_indices = {}
for location, group in items.groupby("item_location_id"):
    location_to_indices[location] = group.index.to_numpy(dtype=np.int64)
print("Строим microcat index...")
microcat_to_indices = defaultdict(list)
for idx, mc in enumerate(item_microcats):
    if mc is None:
        continue
    microcat_to_indices[mc].append(idx)
for mc in list(microcat_to_indices):
    microcat_to_indices[mc] = np.asarray(microcat_to_indices[mc], dtype=np.int64)
print("\nСтроим geo prior...")
geo_counts = (
    supervision.groupby(["search_location_id", "item_location_id"], dropna=False)
    .size()
    .reset_index(name="count")
)
geo_counts["total"] = geo_counts.groupby("search_location_id")["count"].transform("sum")
geo_counts["prob"] = geo_counts["count"] / geo_counts["total"]
geo_counts = geo_counts.sort_values(["search_location_id", "count"], ascending=[True, False])
geo_map = {}
for search_location, group in geo_counts.groupby("search_location_id"):
    values = []
    for row in group.itertuples(index=False):
        if row.item_location_id == search_location:
            continue
        values.append((row.item_location_id, float(row.prob)))
    geo_map[search_location] = values
print("Geo search locations:", len(geo_map))
print("\nЗагружаем BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
model.max_seq_length = 128
query_texts = sample["search_query"].fillna("").astype(str).tolist()
print("\nКодируем validation queries...")
query_embeddings = model.encode(
    query_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
).astype(np.float32)
print("\nGlobal BGE top-500...")
global_bge_top = []
for start in tqdm(range(0, len(sample), 20)):
    end = min(start + 20, len(sample))
    scores = query_embeddings[start:end] @ item_embeddings.T
    k = min(K_GLOBAL, len(items))
    positions = np.argpartition(scores, -k, axis=1)[:, -k:]
    for row in range(end - start):
        idx = positions[row]
        order = np.argsort(scores[row, idx])[::-1]
        global_bge_top.append(idx[order])
print("\nLocal BGE top-500...")
local_bge_top = []
for i in tqdm(range(len(sample))):
    search_location = sample.iloc[i]["search_location_id"]
    local_indices = location_to_indices.get(search_location)
    if local_indices is None or len(local_indices) == 0:
        local_bge_top.append(np.array([], dtype=np.int64))
        continue
    local_scores = item_embeddings[local_indices] @ query_embeddings[i]
    local_bge_top.append(top_k_indices(local_scores, local_indices, K_LOCAL_BGE))
print("\nЗагружаем BM25...")
retriever = bm25s.BM25.load(str(BM25_DIR), load_corpus=False)
stemmer = Stemmer.Stemmer("russian")
bm25_query_texts = (
    sample["search_query"].fillna("").astype(str)
    + " "
    + sample["search_infm_params_text"].fillna("").astype(str)
).tolist()
query_tokens = bm25s.tokenize(bm25_query_texts, stopwords=None, stemmer=stemmer)
print("BM25 top-10000...")
bm25_wide, _ = retriever.retrieve(query_tokens, k=min(K_BM25_WIDE, len(items)))
global_bm25_top = bm25_wide[:, :K_GLOBAL]
print("Local BM25...")
local_bm25_top = []
for i in tqdm(range(len(sample))):
    search_location = sample.iloc[i]["search_location_id"]
    candidates = []
    for idx in bm25_wide[i]:
        idx = int(idx)
        if item_locations[idx] == search_location:
            candidates.append(idx)
            if len(candidates) >= K_LOCAL_BM25:
                break
    local_bm25_top.append(np.asarray(candidates, dtype=np.int64))
print("\nMicrocat classifier...")
vectorizer = joblib.load(MICROCAT_VECTORIZER_PATH)
classifier = joblib.load(MICROCAT_CLASSIFIER_PATH)
X = vectorizer.transform(query_texts)
decision = classifier.decision_function(X)
classes = np.asarray([microcat_key(x) for x in classifier.classes_], dtype=object)
k_mc = min(TOP_MICROCATS, len(classes))
mc_positions = np.argpartition(decision, -k_mc, axis=1)[:, -k_mc:]
predicted_microcats = []
for i in range(len(sample)):
    pos = mc_positions[i]
    order = np.argsort(decision[i, pos])[::-1]
    predicted_microcats.append(classes[pos[order]].tolist())
print("Microcat BGE candidates...")
microcat_bge_top = []
microcat_sets = []
for i in tqdm(range(len(sample))):
    selected = predicted_microcats[i]
    selected_set = set(selected)
    microcat_sets.append(selected_set)
    pools = [microcat_to_indices[mc] for mc in selected if mc in microcat_to_indices]
    if not pools:
        microcat_bge_top.append(np.array([], dtype=np.int64))
        continue
    candidate_indices = np.unique(np.concatenate(pools))
    candidate_scores = item_embeddings[candidate_indices] @ query_embeddings[i]
    microcat_bge_top.append(top_k_indices(candidate_scores, candidate_indices, K_MICROCAT))
print("\nСтроим baseline fusion...")
base_scores = []
baseline_top50 = []
baseline_recalls = []
for i in tqdm(range(len(sample))):
    scores = {}
    add_rrf(scores, global_bm25_top[i], 1.0)
    add_rrf(scores, global_bge_top[i], BGE_GLOBAL_WEIGHT)
    add_rrf(scores, local_bge_top[i], LOCAL_BGE_WEIGHT)
    add_rrf(scores, local_bm25_top[i], LOCAL_BM25_WEIGHT)
    add_rrf(scores, microcat_bge_top[i], MICROCAT_SOURCE_WEIGHT)
    search_location = sample.iloc[i]["search_location_id"]
    selected_microcats = microcat_sets[i]
    for idx in scores:
        if item_locations[idx] == search_location:
            scores[idx] += LOCATION_BONUS
        if item_microcats[idx] in selected_microcats:
            scores[idx] += MICROCAT_BONUS
    ranked = sorted(scores, key=scores.get, reverse=True)
    top50 = ranked[:50]
    baseline_top50.append(top50)
    relevant = sample.iloc[i]["relevant_ids"]
    predicted = set(item_ids[top50])
    recall = len(predicted & relevant) / len(relevant)
    baseline_recalls.append(recall)
    base_scores.append(scores)
baseline_mean = float(np.mean(baseline_recalls))
print("\n" + "=" * 90)
print("INDEPENDENT BASELINE")
print("=" * 90)
print(f"Recall@50: " f"{baseline_mean:.5f} " f"({baseline_mean * 100:.2f}%)")
print("\n" + "=" * 90)
print("INDEPENDENT GEO BONUS GRID")
print("=" * 90)
results = []
for top_n in GEO_TOP_NS:
    for weight in GEO_WEIGHTS:
        recalls = []
        improved = 0
        worse = 0
        same = 0
        total_changed = 0
        for i in range(len(sample)):
            search_location = sample.iloc[i]["search_location_id"]
            geo_values = geo_map.get(search_location, [])[:top_n]
            geo_probabilities = {location: probability for (location, probability) in geo_values}
            scores = dict(base_scores[i])
            for idx in scores:
                item_location = item_locations[idx]
                probability = geo_probabilities.get(item_location)
                if probability is not None:
                    scores[idx] += weight * probability
            ranked = sorted(scores, key=scores.get, reverse=True)
            top50 = ranked[:50]
            old_top = set(baseline_top50[i])
            new_top = set(top50)
            total_changed += len(old_top ^ new_top) // 2
            relevant = sample.iloc[i]["relevant_ids"]
            predicted = set(item_ids[top50])
            recall = len(predicted & relevant) / len(relevant)
            recalls.append(recall)
            old_recall = baseline_recalls[i]
            if recall > old_recall:
                improved += 1
            elif recall < old_recall:
                worse += 1
            else:
                same += 1
        mean_recall = float(np.mean(recalls))
        delta_pp = (mean_recall - baseline_mean) * 100
        avg_changed = total_changed / len(sample)
        results.append(
            {
                "top_n": top_n,
                "weight": weight,
                "recall": mean_recall,
                "delta_pp": delta_pp,
                "improved": improved,
                "worse": worse,
                "same": same,
                "avg_changed": avg_changed,
            }
        )
        print(
            f"topN={top_n:<2} | "
            f"weight={weight:>6.3f} | "
            f"Recall={mean_recall * 100:6.2f}% | "
            f"delta={delta_pp:+.3f} pp | "
            f"+groups={improved:>4} | "
            f"-groups={worse:>4} | "
            f"changed={avg_changed:.2f}"
        )
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(
    ["recall", "improved", "worse"], ascending=[False, False, True]
).reset_index(drop=True)
print("\n" + "=" * 90)
print("BEST INDEPENDENT GEO CONFIGS")
print("=" * 90)
print(
    result_df.head(15).to_string(
        index=False, formatters={"recall": lambda x: f"{x * 100:.3f}%", "delta_pp": lambda x: f"{x:+.3f}"}
    )
)
best = result_df.iloc[0]
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print("Independent baseline:", f"{baseline_mean * 100:.3f}%")
print("Best top_n:", int(best["top_n"]))
print("Best weight:", float(best["weight"]))
print("Best Recall@50:", f"{best['recall'] * 100:.3f}%")
print("Delta:", f"{best['delta_pp']:+.3f} pp")
print("Improved groups:", int(best["improved"]))
print("Worse groups:", int(best["worse"]))
print("Avg changed items/query:", f"{best['avg_changed']:.3f}")
