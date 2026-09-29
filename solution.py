from pathlib import Path
from collections import defaultdict
import gc
import re
import bm25s
import joblib
import numpy as np
import pandas as pd
import Stemmer
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

def select_device():
    """Выбираем самое быстрое доступное устройство, логика поиска от этого не меняется."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
DEVICE = select_device()
ROOT = Path(__file__).resolve().parent
if ROOT.name == "data":
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.parquet"
QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
BGE_EMB_PATH = DATA_DIR / "benchmark_bge_embeddings.npy"
BGE_IDS_PATH = DATA_DIR / "benchmark_bge_item_ids.npy"
TRAIN_BGE_EMB_PATH = DATA_DIR / "train_bge_embeddings.npy"
TRAIN_BGE_IDS_PATH = DATA_DIR / "train_bge_item_ids.npy"
BM25_DIR = DATA_DIR / "benchmark_bm25_index"
MICROCAT_VECTORIZER_PATH = DATA_DIR / "microcat_vectorizer.joblib"
MICROCAT_CLASSIFIER_PATH = DATA_DIR / "microcat_classifier.joblib"
OUTPUT_PATH = ROOT / "answer.csv"
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
ALT_GEO_BGE_WEIGHT = 0.25
ALT_GEO_BM25_WEIGHT = 0.50
K_JOINT_BGE = 500
EXACT_MC_BGE_WEIGHT = 0.50
ALT_MC_BGE_WEIGHT = 0.25
HISTORY_PROTO_TOP_K = 500
HISTORY_PROTO_WEIGHT_DEFAULT = 0.75
HISTORY_PROTO_WEIGHT_SINGLETON = 1.25
BGE_MAX_LENGTH = 128
QUERY_BLOCK_SIZE = 20

def norm_query(value):
    """Одинаково нормализуем текст запроса в обучающей и тестовой выборках, чтобы история точных совпадений работала корректно."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().strip().split())

def microcat_key(value):
    """Приводим идентификатор microcat к единому строковому виду, чтобы разные числовые представления не расходились."""
    if pd.isna(value):
        return None
    try:
        x = float(value)
        if x.is_integer():
            return str(int(x))
    except (TypeError, ValueError):
        pass
    return str(value)

def add_rrf(fusion_scores, indices, weight):
    """Добавляем один ранжированный список кандидатов в общий RRF-балл с заданным весом."""
    for rank, idx in enumerate(indices, start=1):
        idx = int(idx)
        fusion_scores[idx] = fusion_scores.get(idx, 0.0) + weight / (RRF_K + rank)

def sorted_top_k(scores, indices, k):
    """Берём top-k без полной сортировки массива — на больших пулах кандидатов это заметно быстрее."""
    if len(indices) == 0:
        return np.array([], dtype=np.int64)
    k = min(k, len(indices))
    if k == len(indices):
        order = np.argsort(scores)[::-1]
        return indices[order]
    positions = np.argpartition(scores, -k)[-k:]
    order = np.argsort(scores[positions])[::-1]
    return indices[positions[order]]
print("Устройство:", DEVICE)


# Перед запуском поиска проверяю наличие всех заранее подготовленных артефактов.
# Если какого-то файла нет, лучше остановить выполнение сразу, а не получать ошибку уже в середине пайплайна.
required_paths = [
    TRAIN_PATH,
    QUERIES_PATH,
    ITEMS_PATH,
    BGE_EMB_PATH,
    BGE_IDS_PATH,
    TRAIN_BGE_EMB_PATH,
    TRAIN_BGE_IDS_PATH,
    BM25_DIR,
    MICROCAT_VECTORIZER_PATH,
    MICROCAT_CLASSIFIER_PATH,
]
for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Не найден необходимый артефакт: {path}")


# Загружаю тестовые запросы и объявления. Порядок item_id здесь принципиален:
# эмбеддинги и BM25-индексы дальше должны ссылаться на те же позиции в массиве.
print("\nЗагружаем тестовые запросы...")
queries = pd.read_parquet(QUERIES_PATH).reset_index(drop=True)
print("Форма таблицы запросов:", queries.shape)
print("\nЗагружаем тестовые объявления...")
items = pd.read_parquet(ITEMS_PATH).drop_duplicates("item_id").reset_index(drop=True)
print("Форма таблицы объявлений:", items.shape)
item_ids = items["item_id"].astype(str).to_numpy()
item_locations = items["item_location_id"].to_numpy()
item_microcats = np.asarray([microcat_key(x) for x in items["item_microcat_id"]], dtype=object)
benchmark_item_set = set(item_ids.tolist())


# Для семантического поиска использую заранее рассчитанные эмбеддинги объявлений.
# Запросы кодирую той же моделью BGE-M3, после чего близость считаю обычным скалярным произведением.
print("\nЗагружаем BGE-эмбеддинги объявлений...")
item_embeddings = np.load(BGE_EMB_PATH)
saved_item_ids = np.load(BGE_IDS_PATH, allow_pickle=True)
assert len(item_embeddings) == len(items)
assert np.array_equal(saved_item_ids.astype(str), item_ids), (
    "Порядок item_id тестовых объявлений не совпадает с эмбеддингами!"
)
print("Размер эмбеддингов:", item_embeddings.shape)
print("\nЗагружаем BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
model.max_seq_length = BGE_MAX_LENGTH
query_texts = queries["search_query"].fillna("").astype(str).tolist()
print("\nКодируем тестовые запросы...")
query_embeddings = model.encode(
    query_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
).astype(np.float32)


# Первый источник кандидатов — глобальный BGE-поиск по всему корпусу.
# Он хорошо повышает полноту, но сам по себе не учитывает географию и тип услуги.
print("\nГлобальный BGE: выбираю 500 кандидатов...")
global_bge_top = []
for start in tqdm(range(0, len(queries), QUERY_BLOCK_SIZE)):
    end = min(start + QUERY_BLOCK_SIZE, len(queries))
    scores = query_embeddings[start:end] @ item_embeddings.T
    k = min(K_GLOBAL, len(items))
    positions = np.argpartition(scores, -k, axis=1)[:, -k:]
    for row in range(end - start):
        idx = positions[row]
        order = np.argsort(scores[row, idx])[::-1]
        global_bge_top.append(idx[order])


# Отдельно строю индекс объявлений по локации. Это позволяет быстро выполнять локальный поиск
# только внутри нужного города или региона, не проходя каждый раз по всему тестовому корпусу.
print("\nСтроим индекс по локациям...")
location_to_indices = {}
for location, group in items.groupby("item_location_id"):
    location_to_indices[location] = group.index.to_numpy(dtype=np.int64)
print("Локальный BGE: выбираю 500 кандидатов...")
local_bge_top = []
for i in tqdm(range(len(queries))):
    search_location = queries.iloc[i]["search_location_id"]
    local_indices = location_to_indices.get(search_location)
    if local_indices is None or len(local_indices) == 0:
        local_bge_top.append(np.array([], dtype=np.int64))
        continue
    local_scores = item_embeddings[local_indices] @ query_embeddings[i]
    local_bge_top.append(sorted_top_k(local_scores, local_indices, K_LOCAL_BGE))


# BM25 использую как независимый лексический канал. На коротких запросах по услугам
# точные совпадения слов и параметров часто находят релевантные объявления, которые BGE может опустить ниже.
print("\nЗагружаем BM25-индекс тестовых объявлений...")
retriever = bm25s.BM25.load(str(BM25_DIR), load_corpus=False)
stemmer = Stemmer.Stemmer("russian")
bm25_query_texts = (
    queries["search_query"].fillna("").astype(str)
    + " "
    + queries["search_infm_params_text"].fillna("").astype(str)
).tolist()
query_tokens = bm25s.tokenize(bm25_query_texts, stopwords=None, stemmer=stemmer)
print("BM25: выбираю 10000 кандидатов...")
bm25_wide, _ = retriever.retrieve(query_tokens, k=min(K_BM25_WIDE, len(items)))
global_bm25_top = bm25_wide[:, :K_GLOBAL]
print("Локальный BM25...")
local_bm25_top = []
for i in tqdm(range(len(queries))):
    search_location = queries.iloc[i]["search_location_id"]
    candidates = []
    for idx in bm25_wide[i]:
        idx = int(idx)
        if item_locations[idx] == search_location:
            candidates.append(idx)
            if len(candidates) >= K_LOCAL_BM25:
                break
    local_bm25_top.append(np.asarray(candidates, dtype=np.int64))


# Дополнительно предсказываю microcat запроса. Этот сигнал помогает сузить пул кандидатов,
# когда по тексту несколько соседних типов услуг выглядят семантически похожими.
print("\nЗагружаем классификатор microcat...")
vectorizer = joblib.load(MICROCAT_VECTORIZER_PATH)
classifier = joblib.load(MICROCAT_CLASSIFIER_PATH)
query_features = vectorizer.transform(query_texts)
decision = classifier.decision_function(query_features)
classes = np.asarray([microcat_key(x) for x in classifier.classes_], dtype=object)
k_mc = min(TOP_MICROCATS, len(classes))
mc_positions = np.argpartition(decision, -k_mc, axis=1)[:, -k_mc:]
predicted_microcats = []
for i in range(len(queries)):
    pos = mc_positions[i]
    order = np.argsort(decision[i, pos])[::-1]
    predicted_microcats.append(classes[pos[order]].tolist())
print("Строим индекс по microcat...")
microcat_to_indices = defaultdict(list)
for idx, mc in enumerate(item_microcats):
    if mc is None:
        continue
    microcat_to_indices[mc].append(idx)
for mc in list(microcat_to_indices):
    microcat_to_indices[mc] = np.asarray(microcat_to_indices[mc], dtype=np.int64)


# Из обучающей выборки здесь использую только агрегированные исторические сигналы:
# историю точных запросов, распределение microcat и статистику переходов между локациями запроса и объявления.
print("\nЗагружаем историю из обучающей выборки...")
train = pd.read_parquet(
    TRAIN_PATH,
    columns=["search_query", "search_location_id", "item_id", "item_location_id", "item_microcat_id"],
)
train["item_id"] = train["item_id"].astype(str)
train["_norm_query"] = [norm_query(x) for x in train["search_query"]]
train["_mc"] = [microcat_key(x) for x in train["item_microcat_id"]]
print("Строим историю точных совпадений запросов...")
history_in_benchmark = train[train["item_id"].isin(benchmark_item_set)].copy()
hist_counts = (
    history_in_benchmark.groupby(["_norm_query", "item_id"])
    .size()
    .reset_index(name="count")
    .sort_values(["_norm_query", "count"], ascending=[True, False])
)
# Если такой же нормализованный запрос уже встречался в обучающей выборке, сохраняю его реальные релевантные объявления.
# Для повторяющихся запросов это самый прямой сигнал релевантности, поэтому такие item_id ставлю выше обычных кандидатов поиска.
history_items = hist_counts.groupby("_norm_query")["item_id"].apply(list).to_dict()
print("Строим исторические microcat...")
mc_counts = (
    train.dropna(subset=["_mc"])
    .groupby(["_norm_query", "_mc"])
    .size()
    .reset_index(name="count")
    .sort_values(["_norm_query", "count"], ascending=[True, False])
)
history_microcats = {}
for query, group in mc_counts.groupby("_norm_query"):
    history_microcats[query] = group["_mc"].head(5).tolist()


# Географический prior оцениваю по обучающей выборке как частоту item_location для каждой search_location.
# Точное совпадение локации рассматриваю отдельно, а здесь нужны наиболее вероятные альтернативные локации.
print("\nСтроим географический prior...")
geo_counts = (
    train.groupby(["search_location_id", "item_location_id"], dropna=False).size().reset_index(name="count")
)
geo_counts["total"] = geo_counts.groupby("search_location_id")["count"].transform("sum")
geo_counts["prob"] = geo_counts["count"] / geo_counts["total"]
geo_counts = geo_counts.sort_values(["search_location_id", "count"], ascending=[True, False])
# Для каждой search_location оставляю только несколько самых частых альтернативных item_location,
# чтобы географический сигнал не размывался большим количеством редких переходов.
geo_map = {}
for search_location, group in geo_counts.groupby("search_location_id"):
    alternatives = []
    for row in group.itertuples(index=False):
        if row.item_location_id == search_location:
            continue
        alternatives.append((row.item_location_id, float(row.prob)))
    geo_map[search_location] = alternatives[:GEO_TOP_N]
print("Локаций с географическим prior:", len(geo_map))
print("\nBGE-поиск по альтернативным локациям...")
alt_geo_bge_top = []
for i in tqdm(range(len(queries))):
    search_location = queries.iloc[i]["search_location_id"]
    alt_locations = [location for location, _ in geo_map.get(search_location, [])]
    pools = [location_to_indices[location] for location in alt_locations if location in location_to_indices]
    if not pools:
        alt_geo_bge_top.append(np.array([], dtype=np.int64))
        continue
    candidate_indices = np.unique(np.concatenate(pools))
    candidate_scores = item_embeddings[candidate_indices] @ query_embeddings[i]
    alt_geo_bge_top.append(sorted_top_k(candidate_scores, candidate_indices, K_ALT_GEO_BGE))
print("BM25-поиск по альтернативным локациям...")
alt_geo_bm25_top = []
for i in tqdm(range(len(queries))):
    search_location = queries.iloc[i]["search_location_id"]
    alt_location_set = {location for location, _ in geo_map.get(search_location, [])}
    candidates = []
    if alt_location_set:
        for idx in bm25_wide[i]:
            idx = int(idx)
            if item_locations[idx] in alt_location_set:
                candidates.append(idx)
                if len(candidates) >= K_ALT_GEO_BM25:
                    break
    alt_geo_bm25_top.append(np.asarray(candidates, dtype=np.int64))
alt_bge_nonempty = sum(len(x) > 0 for x in alt_geo_bge_top)
alt_bm25_nonempty = sum(len(x) > 0 for x in alt_geo_bm25_top)
print("Запросов с BGE-кандидатами из альтернативных локаций:", alt_bge_nonempty, "/", len(queries))
print("Запросов с BM25-кандидатами из альтернативных локаций:", alt_bm25_nonempty, "/", len(queries))
benchmark_norm_queries = [norm_query(x) for x in queries["search_query"]]
usable_history_queries = sum(q in history_items for q in benchmark_norm_queries)
print("Запросов с доступной историей точного совпадения:", usable_history_queries)


# Для повторяющегося запроса история точных совпадений покрывает только уже виденные item_id, поэтому дополнительно строю прототип истории:
# усредняю эмбеддинги известных релевантных объявлений и по этому вектору ищу похожие объявления в тестовом корпусе.
print("\nСтроим прототипы истории для точных запросов...")
train_item_embeddings = np.load(TRAIN_BGE_EMB_PATH, mmap_mode="r")
train_embedding_ids = np.load(TRAIN_BGE_IDS_PATH, allow_pickle=True)
assert len(train_item_embeddings) == len(train_embedding_ids)
train_embedding_ids = train_embedding_ids.astype(str)
train_item_id_to_embedding_idx = {item_id: idx for idx, item_id in enumerate(train_embedding_ids)}
full_history_counts = train.groupby(["_norm_query", "item_id"]).size().reset_index(name="count")
benchmark_warm_query_set = set(benchmark_norm_queries)
full_history_counts = full_history_counts[full_history_counts["_norm_query"].isin(benchmark_warm_query_set)]
history_count_groups = {query: group for query, group in full_history_counts.groupby("_norm_query")}
warm_query_positions = []
warm_prototype_vectors = []
history_proto_hist_len = np.zeros(len(queries), dtype=np.int16)
for i, nq in enumerate(benchmark_norm_queries):
    group = history_count_groups.get(nq)
    if group is None or len(group) == 0:
        continue
    embedding_indices = []
    counts = []
    for row in group.itertuples(index=False):
        item_id = str(row.item_id)
        embedding_idx = train_item_id_to_embedding_idx.get(item_id)
        if embedding_idx is None:
            continue
        embedding_indices.append(embedding_idx)
        counts.append(float(row.count))
    if not embedding_indices:
        continue
    history_proto_hist_len[i] = len(embedding_indices)
    embedding_indices = np.asarray(embedding_indices, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.float32)
    historical_embeddings = np.asarray(train_item_embeddings[embedding_indices], dtype=np.float32)
    prototype = np.average(historical_embeddings, axis=0, weights=counts).astype(np.float32)
    norm = np.linalg.norm(prototype)
    if norm <= 1e-12:
        continue
    prototype /= norm
    warm_query_positions.append(i)
    warm_prototype_vectors.append(prototype)
print("Повторяющихся запросов с прототипом:", len(warm_query_positions), "/", len(queries))
history_proto_top = [np.array([], dtype=np.int64) for _ in range(len(queries))]
if warm_prototype_vectors:
    warm_prototype_vectors = np.vstack(warm_prototype_vectors).astype(np.float32)
    print("Поиск по прототипам истории...")
    PROTO_BLOCK_SIZE = 20
    for start in tqdm(range(0, len(warm_query_positions), PROTO_BLOCK_SIZE)):
        end = min(start + PROTO_BLOCK_SIZE, len(warm_query_positions))
        scores = warm_prototype_vectors[start:end] @ item_embeddings.T
        k = min(HISTORY_PROTO_TOP_K, len(items))
        positions = np.argpartition(scores, -k, axis=1)[:, -k:]
        for row in range(end - start):
            candidate_indices = positions[row]
            order = np.argsort(scores[row, candidate_indices])[::-1]
            benchmark_query_idx = warm_query_positions[start + row]
            history_proto_top[benchmark_query_idx] = candidate_indices[order].astype(np.int64)
proto_nonempty = sum(len(x) > 0 for x in history_proto_top)
print("Запросов с непустым прототипом истории:", proto_nonempty, "/", len(queries))
del train_item_id_to_embedding_idx
del train_embedding_ids
del full_history_counts
del history_count_groups
gc.collect()


# После глобальных источников добавляю более точные семантические пулы кандидатов:
# отдельно внутри предсказанных microcat и внутри пересечений локация × microcat.
print("\nBGE-кандидаты внутри microcat...")
microcat_bge_top = []
combined_microcat_sets = []
for i in tqdm(range(len(queries))):
    nq = benchmark_norm_queries[i]
    selected = []
    for mc in history_microcats.get(nq, []):
        if mc is not None and mc not in selected:
            selected.append(mc)
    for mc in predicted_microcats[i]:
        if mc is not None and mc not in selected:
            selected.append(mc)
    selected = selected[:10]
    selected_set = set(selected)
    combined_microcat_sets.append(selected_set)
    pools = [microcat_to_indices[mc] for mc in selected if mc in microcat_to_indices]
    if not pools:
        microcat_bge_top.append(np.array([], dtype=np.int64))
        continue
    candidate_indices = np.unique(np.concatenate(pools))
    candidate_scores = item_embeddings[candidate_indices] @ query_embeddings[i]
    microcat_bge_top.append(sorted_top_k(candidate_scores, candidate_indices, K_MICROCAT))
print("\nBGE-кандидаты внутри точной и альтернативной локации × microcat...")
exact_mc_bge_top = []
alt_mc_bge_top = []
for i in tqdm(range(len(queries))):
    search_location = queries.iloc[i]["search_location_id"]
    selected_mc_set = {mc for mc in predicted_microcats[i] if mc is not None}
    pools = [microcat_to_indices[mc] for mc in selected_mc_set if mc in microcat_to_indices]
    if not pools:
        exact_mc_bge_top.append(np.array([], dtype=np.int64))
        alt_mc_bge_top.append(np.array([], dtype=np.int64))
        continue
    mc_indices = np.unique(np.concatenate(pools))
    exact_mask = item_locations[mc_indices] == search_location
    exact_indices = mc_indices[exact_mask]
    if len(exact_indices) == 0:
        exact_mc_bge_top.append(np.array([], dtype=np.int64))
    else:
        exact_scores = item_embeddings[exact_indices] @ query_embeddings[i]
        exact_mc_bge_top.append(sorted_top_k(exact_scores, exact_indices, K_JOINT_BGE))
    alt_locations = {location for location, _ in geo_map.get(search_location, [])}
    if not alt_locations:
        alt_mc_bge_top.append(np.array([], dtype=np.int64))
        continue
    alt_mask = np.isin(item_locations[mc_indices], list(alt_locations))
    alt_indices = mc_indices[alt_mask]
    if len(alt_indices) == 0:
        alt_mc_bge_top.append(np.array([], dtype=np.int64))
    else:
        alt_scores = item_embeddings[alt_indices] @ query_embeddings[i]
        alt_mc_bge_top.append(sorted_top_k(alt_scores, alt_indices, K_JOINT_BGE))
exact_mc_nonempty = sum(len(x) > 0 for x in exact_mc_bge_top)
alt_mc_nonempty = sum(len(x) > 0 for x in alt_mc_bge_top)
print("Запросов с кандидатами точная локация × microcat:", exact_mc_nonempty, "/", len(queries))
print("Запросов с кандидатами альтернативная локация × microcat:", alt_mc_nonempty, "/", len(queries))
assert len(exact_mc_bge_top) == len(queries)
assert len(alt_mc_bge_top) == len(queries)


# Все источники кандидатов объединяю через взвешенный RRF. После этого добавляю небольшие бонусы
# за точное совпадение локации, вероятную географическую альтернативу и подходящий microcat, затем выбираю итоговые 50 объявлений.
print("\nФормируем answer.csv...")
answers = []
history_inserted_total = 0
geo_candidate_boosts = 0
for i in tqdm(range(len(queries))):
    fusion_scores = {}
    add_rrf(fusion_scores, global_bm25_top[i], 1.0)
    add_rrf(fusion_scores, global_bge_top[i], BGE_GLOBAL_WEIGHT)
    add_rrf(fusion_scores, local_bge_top[i], LOCAL_BGE_WEIGHT)
    add_rrf(fusion_scores, local_bm25_top[i], LOCAL_BM25_WEIGHT)
    add_rrf(fusion_scores, alt_geo_bge_top[i], ALT_GEO_BGE_WEIGHT)
    add_rrf(fusion_scores, alt_geo_bm25_top[i], ALT_GEO_BM25_WEIGHT)
    add_rrf(fusion_scores, exact_mc_bge_top[i], EXACT_MC_BGE_WEIGHT)
    add_rrf(fusion_scores, alt_mc_bge_top[i], ALT_MC_BGE_WEIGHT)
    if history_proto_hist_len[i] == 1:
        history_proto_weight = HISTORY_PROTO_WEIGHT_SINGLETON
    else:
        history_proto_weight = HISTORY_PROTO_WEIGHT_DEFAULT
    add_rrf(fusion_scores, history_proto_top[i], history_proto_weight)
    add_rrf(fusion_scores, microcat_bge_top[i], MICROCAT_SOURCE_WEIGHT)
    search_location = queries.iloc[i]["search_location_id"]
    selected_microcats = combined_microcat_sets[i]
    geo_probabilities = {
        location: probability for (location, probability) in geo_map.get(search_location, [])
    }
    for idx in fusion_scores:
        item_location = item_locations[idx]
        if item_location == search_location:
            fusion_scores[idx] += LOCATION_BONUS
        if item_microcats[idx] in selected_microcats:
            fusion_scores[idx] += MICROCAT_BONUS
        geo_probability = geo_probabilities.get(item_location)
        if geo_probability is not None:
            fusion_scores[idx] += GEO_WEIGHT * geo_probability
            geo_candidate_boosts += 1
    ranked_indices = sorted(fusion_scores, key=fusion_scores.get, reverse=True)
    result = []
    seen = set()
    nq = benchmark_norm_queries[i]
    historical_items = history_items.get(nq, [])
    for item_id in historical_items:
        if item_id not in seen and item_id in benchmark_item_set:
            result.append(item_id)
            seen.add(item_id)
            history_inserted_total += 1
            if len(result) >= 50:
                break
    if len(result) < 50:
        for idx in ranked_indices:
            item_id = item_ids[int(idx)]
            if item_id in seen:
                continue
            result.append(item_id)
            seen.add(item_id)
            if len(result) >= 50:
                break
    if len(result) < 50:
        for idx in bm25_wide[i]:
            item_id = item_ids[int(idx)]
            if item_id in seen:
                continue
            result.append(item_id)
            seen.add(item_id)
            if len(result) >= 50:
                break
    answers.append(" ".join(result[:50]))
submission = pd.DataFrame({"query_id": queries["query_id"].astype(str), "answer": answers})


# Перед сохранением проверяю формат ответа: для каждого запроса должно быть ровно 50 уникальных item_id,
# причём каждый id обязан существовать в benchmark_items. Это защищает от незаметных ошибок при формировании answer.csv.
print("\nПроверяем answer.csv...")
assert list(submission.columns) == ["query_id", "answer"]
assert len(submission) == 2452
assert submission["query_id"].nunique() == 2452
assert not (submission["answer"].isna().any())
hex_pattern = re.compile(r"^[0-9a-f]{16}$")
lengths = []
for row_idx, answer in enumerate(submission["answer"]):
    ids = answer.split()
    lengths.append(len(ids))
    assert len(ids) == 50, f"Row {row_idx}: " f"{len(ids)} items"
    assert len(ids) == len(set(ids)), f"Duplicate item IDs " f"in row {row_idx}"
    for item_id in ids:
        assert item_id in benchmark_item_set, f"Unknown item_id: " f"{item_id}"
        assert hex_pattern.fullmatch(item_id), f"Bad item_id: " f"{item_id}"
submission.to_csv(OUTPUT_PATH, index=False)
print("\n" + "=" * 80)
print("ГОТОВО — ИТОГОВЫЙ ОТВЕТ")
print("=" * 80)
print("Число альтернативных локаций:", GEO_TOP_N)
print("Вес географического prior:", GEO_WEIGHT)
print("Добавлено item_id из истории:", history_inserted_total)
print("Применено географических бонусов:", geo_candidate_boosts)
print("Строк в ответе:", len(submission))
print("Минимум кандидатов на запрос:", min(lengths))
print("Максимум кандидатов на запрос:", max(lengths))
print("Файл результата:", OUTPUT_PATH)
print("Размер файла:", OUTPUT_PATH.stat().st_size, "байт")
print("\nПервые 2 строки:")
print(submission.head(2).to_string(index=False))
print("Вес BGE по альтернативным локациям:", ALT_GEO_BGE_WEIGHT)
print("Вес BM25 по альтернативным локациям:", ALT_GEO_BM25_WEIGHT)
print("Вес BGE для точной локации × microcat:", EXACT_MC_BGE_WEIGHT)
print("Вес BGE для альтернативной локации × microcat:", ALT_MC_BGE_WEIGHT)
print("Обычный вес прототипа истории:", HISTORY_PROTO_WEIGHT_DEFAULT)
print("Вес прототипа истории для одного item:", HISTORY_PROTO_WEIGHT_SINGLETON)
print("Запросов с одним item в истории:", int(np.sum(history_proto_hist_len == 1)))
print("Запросов с несколькими item в истории:", int(np.sum(history_proto_hist_len >= 2)))
print("Размер выдачи по прототипу истории:", HISTORY_PROTO_TOP_K)
print("Запросов с прототипом истории:", proto_nonempty)
