from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
answer = pd.read_csv(ROOT / "answer.csv", dtype=str)
queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id"])
assert list(answer.columns) == ["query_id", "answer"]
assert len(answer) == len(queries)
assert answer["query_id"].tolist() == queries["query_id"].astype(str).tolist()
valid_ids = set(items["item_id"].astype(str))
# Отдельно проверяю типичные ошибки итогового файла: неправильное число кандидатов, повторяющиеся ID
# и item_id, которых нет в тестовом корпусе. Такие ошибки могли бы незаметно ухудшить Recall@50.
for row in answer.itertuples(index=False):
    ids = str(row.answer).split()
    assert len(ids) == 50
    assert len(set(ids)) == 50
    assert set(ids).issubset(valid_ids)
print("Формат итогового файла: корректный")
print("Строк:", len(answer))
print("Для каждого запроса указано 50 уникальных корректных item_id.")
