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
DATA = ROOT / "data"

TRAIN_PATH = DATA / "train.parquet"

BGE_EMB_PATH = DATA / "train_bge_embeddings.npy"
BGE_IDS_PATH = DATA / "train_bge_item_ids.npy"

BM25_DIR = DATA / "train_bm25_index"

MC_VECTORIZER_PATH = DATA / "microcat_vectorizer.joblib"
MC_CLASSIFIER_PATH = DATA / "microcat_classifier.joblib"

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
K_JOINT_BM25 = 500

JOINT_BGE_WEIGHTS = [0.0, 0.10, 0.25, 0.50]

JOINT_BM25_WEIGHTS = [0.0, 0.25, 0.50, 0.75]

RANDOM_STATE = 42

TUNE_START = 13000
TUNE_END = 16000

CONFIRM_START = 16000
CONFIRM_END = 19000


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


def recall_at_50(indices, relevant, item_ids):

    predicted = set(item_ids[indices])

    return len(predicted & relevant) / len(relevant)


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

splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_STATE)

train_idx, val_idx = next(splitter.split(train, groups=train["search_query"]))

supervision = train.iloc[train_idx].copy()

holdout = train.iloc[val_idx].copy()

assert not (set(supervision["search_query"]) & set(holdout["search_query"]))

print("Supervision rows:", len(supervision))

print("Holdout rows:", len(holdout))

QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

val_groups = holdout.groupby(QUERY_COLUMNS, dropna=False)["item_id"].agg(set).reset_index(name="relevant_ids")

shuffled = val_groups.sample(frac=1, random_state=123).reset_index(drop=True)

assert len(shuffled) >= CONFIRM_END

sample = shuffled.iloc[TUNE_START:CONFIRM_END].copy().reset_index(drop=True)

N_TUNE = TUNE_END - TUNE_START

N_CONFIRM = CONFIRM_END - CONFIRM_START

print("\nTUNE range:", f"{TUNE_START}:{TUNE_END}")

print("TUNE groups:", N_TUNE)

print("CONFIRM range:", f"{CONFIRM_START}:{CONFIRM_END}")

print("CONFIRM groups:", N_CONFIRM)

print("Total evaluation groups:", len(sample))

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

print("\nItems:", len(items))

print("\nЗагружаем item embeddings...")

item_embeddings = np.load(BGE_EMB_PATH)

saved_ids = np.load(BGE_IDS_PATH, allow_pickle=True)

assert len(item_embeddings) == len(items)

assert np.array_equal(saved_ids.astype(str), item_ids)

print("Embeddings:", item_embeddings.shape)

location_to_indices = {}

for location, group in items.groupby("item_location_id"):

    location_to_indices[location] = group.index.to_numpy(dtype=np.int64)

microcat_to_indices = defaultdict(list)

for idx, mc in enumerate(item_microcats):

    if mc is not None:

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

    alternatives = []

    for row in group.itertuples(index=False):

        if row.item_location_id == search_location:
            continue

        alternatives.append((row.item_location_id, float(row.prob)))

    geo_map[search_location] = alternatives[:GEO_TOP_N]

print("\nЗагружаем BGE-M3...")

model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)

model.max_seq_length = 128

query_texts = sample["search_query"].fillna("").astype(str).tolist()

print("\nКодируем 6000 queries...")

query_embeddings = model.encode(
    query_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
).astype(np.float32)

print("\nGlobal BGE...")

global_bge_top = []

for start in tqdm(range(0, len(sample), 20)):

    end = min(start + 20, len(sample))

    scores = query_embeddings[start:end] @ item_embeddings.T

    pos = np.argpartition(scores, -K_GLOBAL, axis=1)[:, -K_GLOBAL:]

    for row in range(end - start):

        idx = pos[row]

        order = np.argsort(scores[row, idx])[::-1]

        global_bge_top.append(idx[order])

print("Exact/ALT geo BGE...")

local_bge_top = []

alt_geo_bge_top = []

for i in tqdm(range(len(sample))):

    search_location = sample.iloc[i]["search_location_id"]

    exact_indices = location_to_indices.get(search_location)

    if exact_indices is None or len(exact_indices) == 0:

        local_bge_top.append(np.array([], dtype=np.int64))

    else:

        exact_scores = item_embeddings[exact_indices] @ query_embeddings[i]

        local_bge_top.append(top_k_indices(exact_scores, exact_indices, K_LOCAL_BGE))

    alt_locations = [loc for loc, _ in geo_map.get(search_location, [])]

    pools = [location_to_indices[loc] for loc in alt_locations if loc in location_to_indices]

    if not pools:

        alt_geo_bge_top.append(np.array([], dtype=np.int64))

    else:

        alt_indices = np.unique(np.concatenate(pools))

        alt_scores = item_embeddings[alt_indices] @ query_embeddings[i]

        alt_geo_bge_top.append(top_k_indices(alt_scores, alt_indices, K_ALT_GEO_BGE))

print("\nBM25...")

retriever = bm25s.BM25.load(str(BM25_DIR), load_corpus=False)

stemmer = Stemmer.Stemmer("russian")

bm25_query_texts = (
    sample["search_query"].fillna("").astype(str)
    + " "
    + sample["search_infm_params_text"].fillna("").astype(str)
).tolist()

tokens = bm25s.tokenize(bm25_query_texts, stopwords=None, stemmer=stemmer)

bm25_wide, _ = retriever.retrieve(tokens, k=K_BM25_WIDE)

global_bm25_top = bm25_wide[:, :K_GLOBAL]

print("\nMicrocat classifier...")

mc_vectorizer = joblib.load(MC_VECTORIZER_PATH)

mc_classifier = joblib.load(MC_CLASSIFIER_PATH)

X_mc = mc_vectorizer.transform(query_texts)

decision = mc_classifier.decision_function(X_mc)

classes = np.asarray([microcat_key(x) for x in mc_classifier.classes_], dtype=object)

k_mc = min(TOP_MICROCATS, len(classes))

positions = np.argpartition(decision, -k_mc, axis=1)[:, -k_mc:]

predicted_microcats = []

for i in range(len(sample)):

    pos = positions[i]

    order = np.argsort(decision[i, pos])[::-1]

    predicted_microcats.append(classes[pos[order]].tolist())

print("\nСтроим retrieval sources...")

local_bm25_top = []

alt_geo_bm25_top = []

microcat_bge_top = []

exact_mc_bge_top = []

alt_mc_bge_top = []

exact_mc_bm25_top = []

alt_mc_bm25_top = []

microcat_sets = []

for i in tqdm(range(len(sample))):

    search_location = sample.iloc[i]["search_location_id"]

    selected_mc = predicted_microcats[i]

    selected_mc_set = set(selected_mc)

    microcat_sets.append(selected_mc_set)

    alt_location_set = {loc for loc, _ in geo_map.get(search_location, [])}

    local_bm25 = []

    alt_bm25 = []

    exact_joint_bm25 = []

    alt_joint_bm25 = []

    for idx in bm25_wide[i]:

        idx = int(idx)

        loc = item_locations[idx]

        mc = item_microcats[idx]

        if loc == search_location and len(local_bm25) < K_LOCAL_BM25:

            local_bm25.append(idx)

        if loc in alt_location_set and len(alt_bm25) < K_ALT_GEO_BM25:

            alt_bm25.append(idx)

        if loc == search_location and mc in selected_mc_set and len(exact_joint_bm25) < K_JOINT_BM25:

            exact_joint_bm25.append(idx)

        if loc in alt_location_set and mc in selected_mc_set and len(alt_joint_bm25) < K_JOINT_BM25:

            alt_joint_bm25.append(idx)

    local_bm25_top.append(np.asarray(local_bm25, dtype=np.int64))

    alt_geo_bm25_top.append(np.asarray(alt_bm25, dtype=np.int64))

    exact_mc_bm25_top.append(np.asarray(exact_joint_bm25, dtype=np.int64))

    alt_mc_bm25_top.append(np.asarray(alt_joint_bm25, dtype=np.int64))

    pools = [microcat_to_indices[mc] for mc in selected_mc if mc in microcat_to_indices]

    if not pools:

        microcat_bge_top.append(np.array([], dtype=np.int64))

        exact_mc_bge_top.append(np.array([], dtype=np.int64))

        alt_mc_bge_top.append(np.array([], dtype=np.int64))

        continue

    mc_indices = np.unique(np.concatenate(pools))

    mc_scores = item_embeddings[mc_indices] @ query_embeddings[i]

    microcat_bge_top.append(top_k_indices(mc_scores, mc_indices, K_MICROCAT))

    exact_mask = item_locations[mc_indices] == search_location

    exact_joint_indices = mc_indices[exact_mask]

    if len(exact_joint_indices) == 0:

        exact_mc_bge_top.append(np.array([], dtype=np.int64))

    else:

        scores = item_embeddings[exact_joint_indices] @ query_embeddings[i]

        exact_mc_bge_top.append(top_k_indices(scores, exact_joint_indices, K_JOINT_BGE))

    if not alt_location_set:

        alt_mc_bge_top.append(np.array([], dtype=np.int64))

    else:

        alt_mask = np.isin(item_locations[mc_indices], list(alt_location_set))

        alt_joint_indices = mc_indices[alt_mask]

        if len(alt_joint_indices) == 0:

            alt_mc_bge_top.append(np.array([], dtype=np.int64))

        else:

            scores = item_embeddings[alt_joint_indices] @ query_embeddings[i]

            alt_mc_bge_top.append(top_k_indices(scores, alt_joint_indices, K_JOINT_BGE))

print("\nСтроим raw answer3 scores...")

raw_answer3_scores = []

for i in tqdm(range(len(sample))):

    scores = {}

    add_rrf(scores, global_bm25_top[i], 1.0)

    add_rrf(scores, global_bge_top[i], BGE_GLOBAL_WEIGHT)

    add_rrf(scores, local_bge_top[i], LOCAL_BGE_WEIGHT)

    add_rrf(scores, local_bm25_top[i], LOCAL_BM25_WEIGHT)

    add_rrf(scores, alt_geo_bge_top[i], ALT_GEO_BGE_WEIGHT)

    add_rrf(scores, alt_geo_bm25_top[i], ALT_GEO_BM25_WEIGHT)

    add_rrf(scores, microcat_bge_top[i], MICROCAT_SOURCE_WEIGHT)

    raw_answer3_scores.append(scores)


def rank_query(i, exact_bge_weight=0.0, exact_bm25_weight=0.0, alt_bge_weight=0.0, alt_bm25_weight=0.0):

    scores = dict(raw_answer3_scores[i])

    add_rrf(scores, exact_mc_bge_top[i], exact_bge_weight)

    add_rrf(scores, exact_mc_bm25_top[i], exact_bm25_weight)

    add_rrf(scores, alt_mc_bge_top[i], alt_bge_weight)

    add_rrf(scores, alt_mc_bm25_top[i], alt_bm25_weight)

    search_location = sample.iloc[i]["search_location_id"]

    selected_mc = microcat_sets[i]

    geo_prob = {loc: prob for loc, prob in geo_map.get(search_location, [])}

    for idx in scores:

        location = item_locations[idx]

        if location == search_location:

            scores[idx] += LOCATION_BONUS

        if item_microcats[idx] in selected_mc:

            scores[idx] += MICROCAT_BONUS

        probability = geo_prob.get(location)

        if probability is not None:

            scores[idx] += GEO_WEIGHT * probability

    return sorted(scores, key=scores.get, reverse=True)[:50]


def evaluate(
    start,
    end,
    exact_bge_weight=0.0,
    exact_bm25_weight=0.0,
    alt_bge_weight=0.0,
    alt_bm25_weight=0.0,
    baseline_tops=None,
    baseline_recalls=None,
):

    recalls = []

    tops = []

    improved = 0

    worse = 0

    changed_items = 0

    changed_queries = 0

    for i in range(start, end):

        top = rank_query(i, exact_bge_weight, exact_bm25_weight, alt_bge_weight, alt_bm25_weight)

        relevant = sample.iloc[i]["relevant_ids"]

        recall = recall_at_50(top, relevant, item_ids)

        recalls.append(recall)

        tops.append(top)

        if baseline_recalls is not None:

            local_idx = i - start

            old_recall = baseline_recalls[local_idx]

            if recall > old_recall:

                improved += 1

            elif recall < old_recall:

                worse += 1

            old_top = baseline_tops[local_idx]

            changed = len(set(old_top) ^ set(top)) // 2

            changed_items += changed

            if changed > 0:

                changed_queries += 1

    recalls = np.asarray(recalls)

    return {
        "mean": float(recalls.mean()),
        "recalls": recalls,
        "tops": tops,
        "improved": improved,
        "worse": worse,
        "changed_items": changed_items,
        "changed_queries": changed_queries,
    }


print("\n" + "=" * 100)

print("ANSWER3 BASELINE")

print("=" * 100)

tune_base = evaluate(0, N_TUNE)

confirm_base = evaluate(N_TUNE, N_TUNE + N_CONFIRM)

print(f"TUNE answer3: " f"{tune_base['mean'] * 100:.3f}%")

print(f"CONFIRM answer3: " f"{confirm_base['mean'] * 100:.3f}%")


def run_grid(kind):

    rows = []

    print("\n" + "=" * 110)

    print(f"TUNE GRID: {kind}")

    print("=" * 110)

    for bge_weight in JOINT_BGE_WEIGHTS:

        for bm25_weight in JOINT_BM25_WEIGHTS:

            if bge_weight == 0 and bm25_weight == 0:
                continue

            if kind == "EXACT_MC":

                result = evaluate(
                    0,
                    N_TUNE,
                    exact_bge_weight=bge_weight,
                    exact_bm25_weight=bm25_weight,
                    baseline_tops=tune_base["tops"],
                    baseline_recalls=tune_base["recalls"],
                )

            else:

                result = evaluate(
                    0,
                    N_TUNE,
                    alt_bge_weight=bge_weight,
                    alt_bm25_weight=bm25_weight,
                    baseline_tops=tune_base["tops"],
                    baseline_recalls=tune_base["recalls"],
                )

            delta = (result["mean"] - tune_base["mean"]) * 100

            avg_changed = result["changed_items"] / N_TUNE

            rows.append(
                {
                    "bge_weight": bge_weight,
                    "bm25_weight": bm25_weight,
                    "recall": result["mean"],
                    "delta_pp": delta,
                    "improved": result["improved"],
                    "worse": result["worse"],
                    "changed_queries": result["changed_queries"],
                    "avg_changed": avg_changed,
                }
            )

            print(
                f"BGE={bge_weight:>4.2f} | "
                f"BM25={bm25_weight:>4.2f} | "
                f"Recall={result['mean'] * 100:6.3f}% | "
                f"delta={delta:+.3f} | "
                f"+={result['improved']:>3} | "
                f"-={result['worse']:>3} | "
                f"changed={avg_changed:.2f}"
            )

    df = (
        pd.DataFrame(rows)
        .sort_values(["recall", "improved", "worse"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    return df


exact_df = run_grid("EXACT_MC")

print("\nBEST EXACT×MICROCAT:")

print(exact_df.head(10).to_string(index=False))

alt_df = run_grid("ALT_MC")

print("\nBEST ALT×MICROCAT:")

print(alt_df.head(10).to_string(index=False))

best_exact = exact_df.iloc[0]

best_alt = alt_df.iloc[0]

EX_BGE = float(best_exact["bge_weight"])

EX_BM25 = float(best_exact["bm25_weight"])

ALT_BGE = float(best_alt["bge_weight"])

ALT_BM25 = float(best_alt["bm25_weight"])

print("\n" + "=" * 110)

print("INDEPENDENT CONFIRMATION — NEVER USED FOR WEIGHT SELECTION")

print("=" * 110)

confirm_exact = evaluate(
    N_TUNE,
    N_TUNE + N_CONFIRM,
    exact_bge_weight=EX_BGE,
    exact_bm25_weight=EX_BM25,
    baseline_tops=confirm_base["tops"],
    baseline_recalls=confirm_base["recalls"],
)

confirm_alt = evaluate(
    N_TUNE,
    N_TUNE + N_CONFIRM,
    alt_bge_weight=ALT_BGE,
    alt_bm25_weight=ALT_BM25,
    baseline_tops=confirm_base["tops"],
    baseline_recalls=confirm_base["recalls"],
)

tune_combined = evaluate(
    0,
    N_TUNE,
    exact_bge_weight=EX_BGE,
    exact_bm25_weight=EX_BM25,
    alt_bge_weight=ALT_BGE,
    alt_bm25_weight=ALT_BM25,
    baseline_tops=tune_base["tops"],
    baseline_recalls=tune_base["recalls"],
)

confirm_combined = evaluate(
    N_TUNE,
    N_TUNE + N_CONFIRM,
    exact_bge_weight=EX_BGE,
    exact_bm25_weight=EX_BM25,
    alt_bge_weight=ALT_BGE,
    alt_bm25_weight=ALT_BM25,
    baseline_tops=confirm_base["tops"],
    baseline_recalls=confirm_base["recalls"],
)


def print_result(name, result, baseline, n):

    delta = (result["mean"] - baseline["mean"]) * 100

    avg_changed = result["changed_items"] / n

    print(f"\n{name}")

    print(f"Recall: " f"{result['mean'] * 100:.3f}%")

    print(f"Delta: " f"{delta:+.3f} pp")

    print("Improved:", result["improved"])

    print("Worse:", result["worse"])

    print("Changed queries:", result["changed_queries"])

    print("Avg changed:", f"{avg_changed:.3f}")


print("\nSELECTED FROM TUNE:")

print("Exact-MC BGE:", EX_BGE)

print("Exact-MC BM25:", EX_BM25)

print("Alt-MC BGE:", ALT_BGE)

print("Alt-MC BM25:", ALT_BM25)

print_result("CONFIRM EXACT×MICROCAT", confirm_exact, confirm_base, N_CONFIRM)

print_result("CONFIRM ALT×MICROCAT", confirm_alt, confirm_base, N_CONFIRM)

print_result("TUNE COMBINED", tune_combined, tune_base, N_TUNE)

print_result("CONFIRM COMBINED", confirm_combined, confirm_base, N_CONFIRM)

print("\n" + "=" * 110)

print("FINAL SUMMARY")

print("=" * 110)

print("Answer3 TUNE:", f"{tune_base['mean'] * 100:.3f}%")

print("Answer3 CONFIRM:", f"{confirm_base['mean'] * 100:.3f}%")

print("Combined TUNE:", f"{tune_combined['mean'] * 100:.3f}%")

print("Combined CONFIRM:", f"{confirm_combined['mean'] * 100:.3f}%")

print("Combined TUNE delta:", f"{(tune_combined['mean'] - tune_base['mean']) * 100:+.3f} pp")

print("Combined CONFIRM delta:", f"{(confirm_combined['mean'] - confirm_base['mean']) * 100:+.3f} pp")
