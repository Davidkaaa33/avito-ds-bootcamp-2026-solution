from pathlib import Path
from collections import defaultdict
import hashlib
import bm25s
import joblib
import numpy as np
import pandas as pd
import Stemmer
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN_PATH = DATA / "train.parquet"
BGE_EMB_PATH = DATA / "train_bge_embeddings.npy"
BGE_IDS_PATH = DATA / "train_bge_item_ids.npy"
BM25_DIR = DATA / "train_bm25_index"
MC_VEC_PATH = DATA / "microcat_vectorizer.joblib"
MC_CLF_PATH = DATA / "microcat_classifier.joblib"
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
GEO_TOP_N = 3
GEO_WEIGHT = 0.060
ALT_GEO_BGE_WEIGHT = 0.25
ALT_GEO_BM25_WEIGHT = 0.50
K_ALT_GEO_BGE = 500
K_ALT_GEO_BM25 = 500
K_JOINT_BGE = 500
EXACT_MC_BGE_WEIGHT = 0.50
ALT_MC_BGE_WEIGHT = 0.25
PROTO_TOP_K = 500
PROTO_WEIGHTS = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50, 0.60, 0.75]
RANDOM_STATE = 2026
TUNE_START = 4000
TUNE_END = 6000
CONFIRM_START = 6000
CONFIRM_END = 8000

def normalize_query(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().strip().split())

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

def stable_hash(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector
    return vector / norm

def add_rrf(scores, indices, weight):
    if weight == 0:
        return
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
    pos = np.argpartition(scores, -k)[-k:]
    order = np.argsort(scores[pos])[::-1]
    return indices[pos[order]]

def recall_at_50(predicted_indices, target_ids, item_ids):
    predicted = set(item_ids[predicted_indices])
    return len(predicted & target_ids) / len(target_ids)

def build_final_top50(ranked_indices, history_indices):
    result = []
    seen = set()
    for idx in history_indices:
        idx = int(idx)
        if idx in seen:
            continue
        result.append(idx)
        seen.add(idx)
        if len(result) >= 50:
            return np.asarray(result[:50], dtype=np.int64)
    for idx in ranked_indices:
        idx = int(idx)
        if idx in seen:
            continue
        result.append(idx)
        seen.add(idx)
        if len(result) >= 50:
            break
    return np.asarray(result[:50], dtype=np.int64)
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
train["_norm_query"] = [normalize_query(x) for x in train["search_query"]]
train["_mc"] = [microcat_key(x) for x in train["item_microcat_id"]]
print("Rows:", len(train))
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
item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
print("Unique items:", len(items))
print("\nЗагружаем embeddings...")
item_embeddings = np.load(BGE_EMB_PATH).astype(np.float32, copy=False)
saved_ids = np.load(BGE_IDS_PATH, allow_pickle=True)
assert np.array_equal(saved_ids.astype(str), item_ids)
print("Embeddings:", item_embeddings.shape)
print("\nСтроим warm-query universe...")
query_item_counts = train.groupby(["_norm_query", "item_id"]).size().reset_index(name="count")
unique_counts = query_item_counts.groupby("_norm_query")["item_id"].nunique()
warm_queries = unique_counts[unique_counts >= 2].index.tolist()
print("Warm queries:", len(warm_queries))
print("Selecting representative contexts...")
context_counts = (
    train[train["_norm_query"].isin(warm_queries)]
    .groupby(
        ["_norm_query", "search_location_id", "search_infm_params_text", "search_category"], dropna=False
    )
    .size()
    .reset_index(name="context_count")
)
context_counts = context_counts.sort_values(["_norm_query", "context_count"], ascending=[True, False])
context_map = context_counts.drop_duplicates("_norm_query").set_index("_norm_query")
query_text_map = (
    train.drop_duplicates("_norm_query")
    .set_index("_norm_query")["search_query"]
    .fillna("")
    .astype(str)
    .to_dict()
)
print("\nBuilding history / target split...")
query_groups = {
    query: group
    for query, group in query_item_counts[query_item_counts["_norm_query"].isin(warm_queries)].groupby(
        "_norm_query"
    )
}
records = []
for query in tqdm(warm_queries):
    if query not in context_map.index:
        continue
    group = query_groups[query]
    item_count_map = {str(row.item_id): int(row.count) for row in group.itertuples(index=False)}
    unique_ids = list(item_count_map.keys())
    ordered_ids = sorted(unique_ids, key=lambda item_id: stable_hash(query + "|" + item_id))
    n_items = len(ordered_ids)
    n_targets = max(1, int(round(n_items * 0.30)))
    n_targets = min(n_targets, 5)
    n_targets = min(n_targets, n_items - 1)
    target_ids = ordered_ids[:n_targets]
    history_ids = ordered_ids[n_targets:]
    if not history_ids or not target_ids:
        continue
    history_ids = sorted(history_ids, key=lambda item_id: (-item_count_map[item_id], item_id))
    history_indices = np.asarray([item_id_to_index[item_id] for item_id in history_ids], dtype=np.int64)
    history_counts = np.asarray([item_count_map[item_id] for item_id in history_ids], dtype=np.float32)
    ctx = context_map.loc[query]
    records.append(
        {
            "norm_query": query,
            "search_query": query_text_map.get(query, query),
            "search_location_id": ctx["search_location_id"],
            "search_infm_params_text": ctx["search_infm_params_text"],
            "search_category": ctx["search_category"],
            "history_ids": history_ids,
            "history_indices": history_indices,
            "history_counts": history_counts,
            "target_ids": set(target_ids),
            "n_history": len(history_ids),
            "n_target": len(target_ids),
        }
    )
warm_df = pd.DataFrame(records)
warm_df = warm_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print("Usable warm queries:", len(warm_df))
assert len(warm_df) >= CONFIRM_END
sample = warm_df.iloc[TUNE_START:CONFIRM_END].copy().reset_index(drop=True)
N_TUNE = TUNE_END - TUNE_START
N_CONFIRM = CONFIRM_END - CONFIRM_START
print()
print("TUNE warm range:", f"{TUNE_START}:{TUNE_END}")
print("TUNE warm queries:", N_TUNE)
print("CONFIRM warm range:", f"{CONFIRM_START}:{CONFIRM_END}")
print("CONFIRM warm queries:", N_CONFIRM)
print("Median history items:", sample["n_history"].median())
location_to_indices = {}
for location, group in items.groupby("item_location_id"):
    location_to_indices[location] = group.index.to_numpy(dtype=np.int64)
microcat_to_indices = defaultdict(list)
for idx, mc in enumerate(item_microcats):
    if mc is not None:
        microcat_to_indices[mc].append(idx)
for mc in list(microcat_to_indices):
    microcat_to_indices[mc] = np.asarray(microcat_to_indices[mc], dtype=np.int64)
print("\nBuilding geo prior...")
geo_counts = (
    train.groupby(["search_location_id", "item_location_id"], dropna=False).size().reset_index(name="count")
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
    geo_map[search_location] = values[:GEO_TOP_N]
print("\nLoading BGE...")
model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
model.max_seq_length = 128
query_texts = sample["search_query"].fillna("").astype(str).tolist()
query_embeddings = model.encode(
    query_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
).astype(np.float32)
print("\nGlobal BGE...")
global_bge_top = []
for start in tqdm(range(0, len(sample), 20)):
    end = min(start + 20, len(sample))
    scores = query_embeddings[start:end] @ item_embeddings.T
    positions = np.argpartition(scores, -K_GLOBAL, axis=1)[:, -K_GLOBAL:]
    for row in range(end - start):
        idx = positions[row]
        order = np.argsort(scores[row, idx])[::-1]
        global_bge_top.append(idx[order])
print("Local / ALT BGE...")
local_bge_top = []
alt_geo_bge_top = []
for i in tqdm(range(len(sample))):
    search_location = sample.iloc[i]["search_location_id"]
    exact_indices = location_to_indices.get(search_location)
    if exact_indices is None or len(exact_indices) == 0:
        local_bge_top.append(np.array([], dtype=np.int64))
    else:
        scores = item_embeddings[exact_indices] @ query_embeddings[i]
        local_bge_top.append(top_k_indices(scores, exact_indices, K_LOCAL_BGE))
    alt_locations = [location for location, _ in geo_map.get(search_location, [])]
    pools = [location_to_indices[location] for location in alt_locations if location in location_to_indices]
    if not pools:
        alt_geo_bge_top.append(np.array([], dtype=np.int64))
    else:
        alt_indices = np.unique(np.concatenate(pools))
        scores = item_embeddings[alt_indices] @ query_embeddings[i]
        alt_geo_bge_top.append(top_k_indices(scores, alt_indices, K_ALT_GEO_BGE))
print("\nBM25...")
retriever = bm25s.BM25.load(str(BM25_DIR), load_corpus=False)
stemmer = Stemmer.Stemmer("russian")
bm25_query_text = (
    sample["search_query"].fillna("").astype(str)
    + " "
    + sample["search_infm_params_text"].fillna("").astype(str)
).tolist()
tokens = bm25s.tokenize(bm25_query_text, stopwords=None, stemmer=stemmer)
bm25_wide, _ = retriever.retrieve(tokens, k=K_BM25_WIDE)
global_bm25_top = bm25_wide[:, :K_GLOBAL]
print("Local / ALT BM25...")
local_bm25_top = []
alt_geo_bm25_top = []
for i in tqdm(range(len(sample))):
    search_location = sample.iloc[i]["search_location_id"]
    alt_set = {location for location, _ in geo_map.get(search_location, [])}
    exact = []
    alt = []
    for idx in bm25_wide[i]:
        idx = int(idx)
        location = item_locations[idx]
        if location == search_location and len(exact) < K_LOCAL_BM25:
            exact.append(idx)
        if location in alt_set and len(alt) < K_ALT_GEO_BM25:
            alt.append(idx)
        if len(exact) >= K_LOCAL_BM25 and len(alt) >= K_ALT_GEO_BM25:
            break
    local_bm25_top.append(np.asarray(exact, dtype=np.int64))
    alt_geo_bm25_top.append(np.asarray(alt, dtype=np.int64))
print("\nMicrocat classifier...")
mc_vectorizer = joblib.load(MC_VEC_PATH)
mc_classifier = joblib.load(MC_CLF_PATH)
X = mc_vectorizer.transform(query_texts)
decision = mc_classifier.decision_function(X)
classes = np.asarray([microcat_key(x) for x in mc_classifier.classes_], dtype=object)
positions = np.argpartition(decision, -TOP_MICROCATS, axis=1)[:, -TOP_MICROCATS:]
predicted_microcats = []
for i in range(len(sample)):
    pos = positions[i]
    order = np.argsort(decision[i, pos])[::-1]
    predicted_microcats.append(classes[pos[order]].tolist())
history_microcat_sets = []
for i in range(len(sample)):
    history_indices = sample.iloc[i]["history_indices"]
    mcs = [item_microcats[idx] for idx in history_indices if item_microcats[idx] is not None]
    history_microcat_sets.append(set(mcs))
print("\nMicrocat / joint BGE...")
microcat_bge_top = []
exact_mc_bge_top = []
alt_mc_bge_top = []
combined_mc_sets = []
for i in tqdm(range(len(sample))):
    classifier_mcs = [mc for mc in predicted_microcats[i] if mc is not None]
    combined = []
    for mc in history_microcat_sets[i]:
        if mc not in combined:
            combined.append(mc)
    for mc in classifier_mcs:
        if mc not in combined:
            combined.append(mc)
    combined = combined[:10]
    combined_set = set(combined)
    combined_mc_sets.append(combined_set)
    pools = [microcat_to_indices[mc] for mc in combined if mc in microcat_to_indices]
    if not pools:
        microcat_bge_top.append(np.array([], dtype=np.int64))
    else:
        indices = np.unique(np.concatenate(pools))
        scores = item_embeddings[indices] @ query_embeddings[i]
        microcat_bge_top.append(top_k_indices(scores, indices, K_MICROCAT))
    joint_pools = [microcat_to_indices[mc] for mc in classifier_mcs if mc in microcat_to_indices]
    if not joint_pools:
        empty = np.array([], dtype=np.int64)
        exact_mc_bge_top.append(empty)
        alt_mc_bge_top.append(empty)
        continue
    joint_indices = np.unique(np.concatenate(joint_pools))
    search_location = sample.iloc[i]["search_location_id"]
    exact_mask = item_locations[joint_indices] == search_location
    exact_indices = joint_indices[exact_mask]
    if len(exact_indices):
        scores = item_embeddings[exact_indices] @ query_embeddings[i]
        exact_mc_bge_top.append(top_k_indices(scores, exact_indices, K_JOINT_BGE))
    else:
        exact_mc_bge_top.append(np.array([], dtype=np.int64))
    alt_set = {location for location, _ in geo_map.get(search_location, [])}
    alt_mask = np.isin(item_locations[joint_indices], list(alt_set))
    alt_indices = joint_indices[alt_mask]
    if len(alt_indices):
        scores = item_embeddings[alt_indices] @ query_embeddings[i]
        alt_mc_bge_top.append(top_k_indices(scores, alt_indices, K_JOINT_BGE))
    else:
        alt_mc_bge_top.append(np.array([], dtype=np.int64))
print("\nHistory prototype retrieval...")
prototype_vectors = []
for i in tqdm(range(len(sample))):
    indices = sample.iloc[i]["history_indices"]
    counts = sample.iloc[i]["history_counts"]
    embeddings = item_embeddings[indices]
    vector = np.average(embeddings, axis=0, weights=counts).astype(np.float32)
    vector = normalize_vector(vector).astype(np.float32)
    prototype_vectors.append(vector)
prototype_vectors = np.vstack(prototype_vectors).astype(np.float32)
prototype_top = []
for start in tqdm(range(0, len(sample), 20), desc="Prototype retrieval"):
    end = min(start + 20, len(sample))
    scores = prototype_vectors[start:end] @ item_embeddings.T
    k = min(PROTO_TOP_K, len(items))
    positions = np.argpartition(scores, -k, axis=1)[:, -k:]
    for row in range(end - start):
        idx = positions[row]
        order = np.argsort(scores[row, idx])[::-1]
        prototype_top.append(idx[order].astype(np.int64))
print("\nBuilding answer4 raw scores...")
raw_answer4_scores = []
for i in tqdm(range(len(sample))):
    scores = {}
    add_rrf(scores, global_bm25_top[i], 1.0)
    add_rrf(scores, global_bge_top[i], BGE_GLOBAL_WEIGHT)
    add_rrf(scores, local_bge_top[i], LOCAL_BGE_WEIGHT)
    add_rrf(scores, local_bm25_top[i], LOCAL_BM25_WEIGHT)
    add_rrf(scores, alt_geo_bge_top[i], ALT_GEO_BGE_WEIGHT)
    add_rrf(scores, alt_geo_bm25_top[i], ALT_GEO_BM25_WEIGHT)
    add_rrf(scores, microcat_bge_top[i], MICROCAT_SOURCE_WEIGHT)
    add_rrf(scores, exact_mc_bge_top[i], EXACT_MC_BGE_WEIGHT)
    add_rrf(scores, alt_mc_bge_top[i], ALT_MC_BGE_WEIGHT)
    raw_answer4_scores.append(scores)

def rank_query(i, prototype_weight):
    scores = dict(raw_answer4_scores[i])
    if prototype_weight > 0:
        add_rrf(scores, prototype_top[i], prototype_weight)
    search_location = sample.iloc[i]["search_location_id"]
    selected_mc_set = combined_mc_sets[i]
    geo_prob = {location: probability for (location, probability) in geo_map.get(search_location, [])}
    for idx in scores:
        location = item_locations[idx]
        if location == search_location:
            scores[idx] += LOCATION_BONUS
        if item_microcats[idx] in selected_mc_set:
            scores[idx] += MICROCAT_BONUS
        probability = geo_prob.get(location)
        if probability is not None:
            scores[idx] += GEO_WEIGHT * probability
    ranked = sorted(scores, key=scores.get, reverse=True)
    final_top = build_final_top50(ranked, sample.iloc[i]["history_indices"])
    return final_top

def evaluate(start, end, prototype_weight, baseline=None):
    recalls = []
    tops = []
    improved = 0
    worse = 0
    changed_queries = 0
    changed_items = 0
    for i in range(start, end):
        top = rank_query(i, prototype_weight)
        recall = recall_at_50(top, sample.iloc[i]["target_ids"], item_ids)
        recalls.append(recall)
        tops.append(top)
        if baseline is not None:
            j = i - start
            old_recall = baseline["recalls"][j]
            if recall > old_recall:
                improved += 1
            elif recall < old_recall:
                worse += 1
            changed = len(set(top) ^ set(baseline["tops"][j])) // 2
            changed_items += changed
            if changed > 0:
                changed_queries += 1
    recalls = np.asarray(recalls, dtype=np.float32)
    return {
        "mean": float(recalls.mean()),
        "recalls": recalls,
        "tops": tops,
        "improved": improved,
        "worse": worse,
        "changed_queries": changed_queries,
        "changed_items": changed_items,
    }
print()
print("=" * 110)
print("FULL ANSWER4 — WARM BASELINE")
print("=" * 110)
tune_base = evaluate(0, N_TUNE, 0.0)
confirm_base = evaluate(N_TUNE, N_TUNE + N_CONFIRM, 0.0)
print("TUNE answer4:", f"{tune_base['mean'] * 100:.3f}%")
print("CONFIRM answer4:", f"{confirm_base['mean'] * 100:.3f}%")
print()
print("=" * 110)
print("HISTORY PROTOTYPE ON TOP OF ANSWER4 — TUNE")
print("=" * 110)
rows = []
for weight in PROTO_WEIGHTS:
    result = evaluate(0, N_TUNE, weight, baseline=tune_base)
    delta = (result["mean"] - tune_base["mean"]) * 100
    avg_changed = result["changed_items"] / N_TUNE
    rows.append(
        {
            "weight": weight,
            "recall": result["mean"],
            "delta_pp": delta,
            "improved": result["improved"],
            "worse": result["worse"],
            "changed_queries": result["changed_queries"],
            "avg_changed": avg_changed,
        }
    )
    print(
        f"weight={weight:>4.2f} | "
        f"Recall={result['mean'] * 100:6.3f}% | "
        f"delta={delta:+.3f} pp | "
        f"+={result['improved']:>4} | "
        f"-={result['worse']:>4} | "
        f"changed_q={result['changed_queries']:>4} | "
        f"avg_changed={avg_changed:.2f}"
    )
result_df = (
    pd.DataFrame(rows)
    .sort_values(["recall", "improved", "worse", "avg_changed"], ascending=[False, False, True, True])
    .reset_index(drop=True)
)
print()
print("=" * 110)
print("BEST TUNE CONFIGS")
print("=" * 110)
print(
    result_df.to_string(
        index=False,
        formatters={
            "recall": lambda x: f"{x * 100:.3f}%",
            "delta_pp": lambda x: f"{x:+.3f}",
            "avg_changed": lambda x: f"{x:.3f}",
        },
    )
)
best = result_df.iloc[0]
BEST_WEIGHT = float(best["weight"])
confirm = evaluate(N_TUNE, N_TUNE + N_CONFIRM, BEST_WEIGHT, baseline=confirm_base)
confirm_delta = (confirm["mean"] - confirm_base["mean"]) * 100
proto_hit_50 = 0
proto_hit_100 = 0
proto_hit_500 = 0
for i in range(N_TUNE, N_TUNE + N_CONFIRM):
    targets = sample.iloc[i]["target_ids"]
    ranking = prototype_top[i]
    if set(item_ids[ranking[:50]]) & targets:
        proto_hit_50 += 1
    if set(item_ids[ranking[:100]]) & targets:
        proto_hit_100 += 1
    if set(item_ids[ranking[:500]]) & targets:
        proto_hit_500 += 1
print()
print("=" * 110)
print("INDEPENDENT WARM CONFIRMATION")
print("=" * 110)
print("Selected prototype weight:", BEST_WEIGHT)
print()
print("ANSWER4 warm baseline:", f"{confirm_base['mean'] * 100:.3f}%")
print("ANSWER4 + prototype:", f"{confirm['mean'] * 100:.3f}%")
print("CONFIRM delta:", f"{confirm_delta:+.3f} pp")
print("Improved warm queries:", confirm["improved"])
print("Worse warm queries:", confirm["worse"])
print("Changed warm queries:", confirm["changed_queries"])
print("Avg changed items/query:", f"{confirm['changed_items'] / N_CONFIRM:.3f}")
print()
print("Prototype hit@50:", f"{proto_hit_50 / N_CONFIRM * 100:.2f}%")
print("Prototype hit@100:", f"{proto_hit_100 / N_CONFIRM * 100:.2f}%")
print("Prototype hit@500:", f"{proto_hit_500 / N_CONFIRM * 100:.2f}%")
print()
print("=" * 110)
print("FINAL SUMMARY")
print("=" * 110)
print("Answer4 TUNE:", f"{tune_base['mean'] * 100:.3f}%")
print("Best TUNE:", f"{best['recall'] * 100:.3f}%")
print("TUNE delta:", f"{best['delta_pp']:+.3f} pp")
print("Answer4 CONFIRM:", f"{confirm_base['mean'] * 100:.3f}%")
print("New CONFIRM:", f"{confirm['mean'] * 100:.3f}%")
print("CONFIRM delta:", f"{confirm_delta:+.3f} pp")
