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

GEO_TOP_N = 3
GEO_WEIGHT = 0.060

K_ALT_GEO_BGE = 500
K_ALT_GEO_BM25 = 500

ALT_BGE_WEIGHTS = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

ALT_BM25_WEIGHTS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
]

RANDOM_STATE = 42

FRESH_START = 10000
FRESH_END = 13000


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


def add_rrf(
    scores,
    indices,
    weight,
):

    if weight == 0:
        return

    for rank, idx in enumerate(
        indices,
        start=1,
    ):

        idx = int(idx)

        scores[idx] = scores.get(
            idx,
            0.0,
        ) + weight / (RRF_K + rank)


def top_k_indices(
    scores,
    indices,
    k,
):

    if len(indices) == 0:

        return np.array(
            [],
            dtype=np.int64,
        )

    k = min(
        k,
        len(indices),
    )

    if k == len(indices):

        order = np.argsort(scores)[::-1]

        return indices[order]

    pos = np.argpartition(
        scores,
        -k,
    )[-k:]

    order = np.argsort(scores[pos])[::-1]

    return indices[pos[order]]


def recall_at_50(
    top_indices,
    relevant,
    item_ids,
):

    predicted = set(item_ids[top_indices])

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

print(
    "Rows:",
    len(train),
)

print("\nСтроим query-disjoint split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=RANDOM_STATE,
)

train_idx, val_idx = next(
    splitter.split(
        train,
        groups=train["search_query"],
    )
)

supervision = train.iloc[train_idx].copy()

holdout = train.iloc[val_idx].copy()

assert not (set(supervision["search_query"]) & set(holdout["search_query"]))

print(
    "Supervision rows:",
    len(supervision),
)

print(
    "Holdout rows:",
    len(holdout),
)

QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

val_groups = (
    holdout.groupby(
        QUERY_COLUMNS,
        dropna=False,
    )["item_id"]
    .agg(set)
    .reset_index(name="relevant_ids")
)

shuffled = val_groups.sample(
    frac=1,
    random_state=123,
).reset_index(drop=True)

assert len(shuffled) >= FRESH_END

sample = shuffled.iloc[FRESH_START:FRESH_END].copy().reset_index(drop=True)

print(
    "Fresh validation range:",
    f"{FRESH_START}:{FRESH_END}",
)

print(
    "Fresh validation groups:",
    len(sample),
)

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

item_microcats = np.asarray(
    [microcat_key(x) for x in items["item_microcat_id"]],
    dtype=object,
)

print(
    "Items:",
    len(items),
)

print("\nЗагружаем BGE embeddings...")

item_embeddings = np.load(BGE_EMB_PATH)

saved_ids = np.load(
    BGE_IDS_PATH,
    allow_pickle=True,
)

assert len(item_embeddings) == len(items)

assert np.array_equal(
    saved_ids.astype(str),
    item_ids,
)

print("Embeddings:", item_embeddings.shape)

print("\nСтроим location index...")

location_to_indices = {}

for location, group in items.groupby("item_location_id"):

    location_to_indices[location] = group.index.to_numpy(dtype=np.int64)

print("Строим microcat index...")

microcat_to_indices = defaultdict(list)

for idx, mc in enumerate(item_microcats):

    if mc is not None:

        microcat_to_indices[mc].append(idx)

for mc in list(microcat_to_indices):

    microcat_to_indices[mc] = np.asarray(
        microcat_to_indices[mc],
        dtype=np.int64,
    )

print("\nСтроим answer2 geo prior...")

geo_counts = (
    supervision.groupby(
        [
            "search_location_id",
            "item_location_id",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="count")
)

geo_counts["total"] = geo_counts.groupby("search_location_id")["count"].transform("sum")

geo_counts["prob"] = geo_counts["count"] / geo_counts["total"]

geo_counts = geo_counts.sort_values(
    [
        "search_location_id",
        "count",
    ],
    ascending=[
        True,
        False,
    ],
)

geo_map = {}

for search_location, group in geo_counts.groupby("search_location_id"):

    alternatives = []

    for row in group.itertuples(index=False):

        if row.item_location_id == search_location:
            continue

        alternatives.append(
            (
                row.item_location_id,
                float(row.prob),
            )
        )

    geo_map[search_location] = alternatives[:GEO_TOP_N]

print("\nЗагружаем BGE-M3...")

model = SentenceTransformer(
    "BAAI/bge-m3",
    device=DEVICE,
)

model.max_seq_length = 128

query_texts = sample["search_query"].fillna("").astype(str).tolist()

print("\nКодируем queries...")

query_embeddings = model.encode(
    query_texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)

print("\nGlobal BGE top-500...")

global_bge_top = []

for start in tqdm(
    range(
        0,
        len(sample),
        20,
    )
):

    end = min(
        start + 20,
        len(sample),
    )

    scores = query_embeddings[start:end] @ item_embeddings.T

    pos = np.argpartition(
        scores,
        -K_GLOBAL,
        axis=1,
    )[:, -K_GLOBAL:]

    for row in range(end - start):

        idx = pos[row]

        order = np.argsort(
            scores[
                row,
                idx,
            ]
        )[::-1]

        global_bge_top.append(idx[order])

print("Exact-location BGE...")

local_bge_top = []

for i in tqdm(range(len(sample))):

    search_location = sample.iloc[i]["search_location_id"]

    indices = location_to_indices.get(search_location)

    if indices is None or len(indices) == 0:

        local_bge_top.append(
            np.array(
                [],
                dtype=np.int64,
            )
        )

        continue

    scores = item_embeddings[indices] @ query_embeddings[i]

    local_bge_top.append(
        top_k_indices(
            scores,
            indices,
            K_LOCAL_BGE,
        )
    )

print("\nAlternative-GEO BGE...")

alt_geo_bge_top = []

for i in tqdm(range(len(sample))):

    search_location = sample.iloc[i]["search_location_id"]

    alt_locations = [
        loc
        for loc, _ in geo_map.get(
            search_location,
            [],
        )
    ]

    pools = [location_to_indices[loc] for loc in alt_locations if loc in location_to_indices]

    if not pools:

        alt_geo_bge_top.append(
            np.array(
                [],
                dtype=np.int64,
            )
        )

        continue

    alt_indices = np.unique(np.concatenate(pools))

    alt_scores = item_embeddings[alt_indices] @ query_embeddings[i]

    alt_geo_bge_top.append(
        top_k_indices(
            alt_scores,
            alt_indices,
            K_ALT_GEO_BGE,
        )
    )

print("\nЗагружаем BM25...")

retriever = bm25s.BM25.load(
    str(BM25_DIR),
    load_corpus=False,
)

stemmer = Stemmer.Stemmer("russian")

bm25_query_texts = (
    sample["search_query"].fillna("").astype(str)
    + " "
    + sample["search_infm_params_text"].fillna("").astype(str)
).tolist()

tokens = bm25s.tokenize(
    bm25_query_texts,
    stopwords=None,
    stemmer=stemmer,
)

print("BM25 top-10000...")

bm25_wide, _ = retriever.retrieve(
    tokens,
    k=K_BM25_WIDE,
)

global_bm25_top = bm25_wide[:, :K_GLOBAL]

print("Формируем exact/alt GEO BM25...")

local_bm25_top = []

alt_geo_bm25_top = []

for i in tqdm(range(len(sample))):

    search_location = sample.iloc[i]["search_location_id"]

    alt_location_set = set(
        loc
        for loc, _ in geo_map.get(
            search_location,
            [],
        )
    )

    exact_candidates = []

    alt_candidates = []

    for idx in bm25_wide[i]:

        idx = int(idx)

        item_location = item_locations[idx]

        if item_location == search_location and len(exact_candidates) < K_LOCAL_BM25:

            exact_candidates.append(idx)

        if item_location in alt_location_set and len(alt_candidates) < K_ALT_GEO_BM25:

            alt_candidates.append(idx)

        if len(exact_candidates) >= K_LOCAL_BM25 and len(alt_candidates) >= K_ALT_GEO_BM25:
            break

    local_bm25_top.append(
        np.asarray(
            exact_candidates,
            dtype=np.int64,
        )
    )

    alt_geo_bm25_top.append(
        np.asarray(
            alt_candidates,
            dtype=np.int64,
        )
    )

print("\nMicrocat classifier...")

vectorizer = joblib.load(MICROCAT_VECTORIZER_PATH)

classifier = joblib.load(MICROCAT_CLASSIFIER_PATH)

X = vectorizer.transform(query_texts)

decision = classifier.decision_function(X)

classes = np.asarray(
    [microcat_key(x) for x in classifier.classes_],
    dtype=object,
)

k_mc = min(
    TOP_MICROCATS,
    len(classes),
)

positions = np.argpartition(
    decision,
    -k_mc,
    axis=1,
)[:, -k_mc:]

predicted_microcats = []

for i in range(len(sample)):

    pos = positions[i]

    order = np.argsort(
        decision[
            i,
            pos,
        ]
    )[::-1]

    predicted_microcats.append(classes[pos[order]].tolist())

print("Microcat BGE...")

microcat_bge_top = []

microcat_sets = []

for i in tqdm(range(len(sample))):

    selected = predicted_microcats[i]

    microcat_sets.append(set(selected))

    pools = [microcat_to_indices[mc] for mc in selected if mc in microcat_to_indices]

    if not pools:

        microcat_bge_top.append(
            np.array(
                [],
                dtype=np.int64,
            )
        )

        continue

    indices = np.unique(np.concatenate(pools))

    scores = item_embeddings[indices] @ query_embeddings[i]

    microcat_bge_top.append(
        top_k_indices(
            scores,
            indices,
            K_MICROCAT,
        )
    )

print("\nСтроим answer2 baseline...")

answer2_scores = []

answer2_top = []

answer2_recalls = []

for i in tqdm(range(len(sample))):

    scores = {}

    add_rrf(
        scores,
        global_bm25_top[i],
        1.0,
    )

    add_rrf(
        scores,
        global_bge_top[i],
        BGE_GLOBAL_WEIGHT,
    )

    add_rrf(
        scores,
        local_bge_top[i],
        LOCAL_BGE_WEIGHT,
    )

    add_rrf(
        scores,
        local_bm25_top[i],
        LOCAL_BM25_WEIGHT,
    )

    add_rrf(
        scores,
        microcat_bge_top[i],
        MICROCAT_SOURCE_WEIGHT,
    )

    search_location = sample.iloc[i]["search_location_id"]

    selected_microcats = microcat_sets[i]

    geo_prob = {
        loc: prob
        for loc, prob in geo_map.get(
            search_location,
            [],
        )
    }

    for idx in scores:

        item_location = item_locations[idx]

        if item_location == search_location:

            scores[idx] += LOCATION_BONUS

        if item_microcats[idx] in selected_microcats:

            scores[idx] += MICROCAT_BONUS

        probability = geo_prob.get(item_location)

        if probability is not None:

            scores[idx] += GEO_WEIGHT * probability

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:50]

    relevant = sample.iloc[i]["relevant_ids"]

    recall = recall_at_50(
        ranked,
        relevant,
        item_ids,
    )

    answer2_scores.append(scores)

    answer2_top.append(ranked)

    answer2_recalls.append(recall)

answer2_recalls = np.asarray(answer2_recalls)

answer2_mean = float(answer2_recalls.mean())

print("\n" + "=" * 100)

print("CURRENT ANSWER2 PIPELINE")

print("=" * 100)

print(f"Recall@50: " f"{answer2_mean * 100:.3f}%")

print("\n" + "=" * 115)

print("ALT-GEO RETRIEVAL GRID")

print("=" * 115)

results = []

for bge_weight in ALT_BGE_WEIGHTS:

    for bm25_weight in ALT_BM25_WEIGHTS:

        if bge_weight == 0 and bm25_weight == 0:
            continue

        recalls = []

        improved = 0

        worse = 0

        total_changed = 0

        changed_queries = 0

        for i in range(len(sample)):

            scores = dict(answer2_scores[i])

            add_rrf(
                scores,
                alt_geo_bge_top[i],
                bge_weight,
            )

            add_rrf(
                scores,
                alt_geo_bm25_top[i],
                bm25_weight,
            )

            ranked = sorted(
                scores,
                key=scores.get,
                reverse=True,
            )[:50]

            relevant = sample.iloc[i]["relevant_ids"]

            recall = recall_at_50(
                ranked,
                relevant,
                item_ids,
            )

            recalls.append(recall)

            old_recall = answer2_recalls[i]

            if recall > old_recall:

                improved += 1

            elif recall < old_recall:

                worse += 1

            changed = len(set(answer2_top[i]) ^ set(ranked)) // 2

            total_changed += changed

            if changed > 0:

                changed_queries += 1

        mean_recall = float(np.mean(recalls))

        delta = (mean_recall - answer2_mean) * 100

        avg_changed = total_changed / len(sample)

        results.append(
            {
                "bge_weight": bge_weight,
                "bm25_weight": bm25_weight,
                "recall": mean_recall,
                "delta_pp": delta,
                "improved": improved,
                "worse": worse,
                "changed_queries": changed_queries,
                "avg_changed": avg_changed,
            }
        )

        print(
            f"BGE={bge_weight:>4.2f} | "
            f"BM25={bm25_weight:>4.2f} | "
            f"Recall={mean_recall * 100:6.3f}% | "
            f"delta={delta:+.3f} pp | "
            f"+={improved:>3} | "
            f"-={worse:>3} | "
            f"changed_q={changed_queries:>4} | "
            f"avg_changed={avg_changed:.2f}"
        )

result_df = (
    pd.DataFrame(results)
    .sort_values(
        [
            "recall",
            "improved",
            "worse",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 115)

print("BEST ALT-GEO CONFIGS")

print("=" * 115)

print(
    result_df.head(20).to_string(
        index=False,
        formatters={
            "recall": lambda x: f"{x * 100:.3f}%",
            "delta_pp": lambda x: f"{x:+.3f}",
            "avg_changed": lambda x: f"{x:.3f}",
        },
    )
)

best = result_df.iloc[0]

print("\n" + "=" * 90)

print("SUMMARY")

print("=" * 90)

print(
    "Fresh sample:",
    f"{FRESH_START}:{FRESH_END}",
)

print("Answer2 baseline:", f"{answer2_mean * 100:.3f}%")

print("Best ALT BGE weight:", float(best["bge_weight"]))

print("Best ALT BM25 weight:", float(best["bm25_weight"]))

print("Best Recall:", f"{best['recall'] * 100:.3f}%")

print("Increment over answer2:", f"{best['delta_pp']:+.3f} pp")

print("Improved groups:", int(best["improved"]))

print("Worse groups:", int(best["worse"]))

print("Changed queries:", int(best["changed_queries"]))

print("Average changed items/query:", f"{best['avg_changed']:.3f}")
