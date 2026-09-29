from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "train.parquet"
VECTORIZER_PATH = DATA_DIR / "microcat_vectorizer.joblib"
CLASSIFIER_PATH = DATA_DIR / "microcat_classifier.joblib"

def build_vectorizer():
    """Объединяем словные и символьные n-граммы: на коротких запросах по услугам вместе они работают стабильнее."""
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80_000, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=120_000, sublinear_tf=True
                ),
            ),
        ]
    )

def evaluate_top_k(classifier, vectorizer, val_df):
    """Проверяем покрытие среди первых k предсказаний, чтобы выбрать разумное число microcat для поиска кандидатов."""
    val_queries = (
        val_df.groupby("search_query")["item_microcat_id"].agg(set).reset_index(name="true_microcats")
    )
    X_val = vectorizer.transform(val_queries["search_query"])
    scores = classifier.decision_function(X_val)
    classes = classifier.classes_
    print("Запросов в валидационной части:", len(val_queries))
    for k in [1, 2, 3, 5, 10]:
        top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
        recalls, hits = [], []
        for i, row in val_queries.iterrows():
            predicted = set(classes[top_indices[i]])
            truth = row["true_microcats"]
            recalls.append(len(predicted & truth) / len(truth))
            hits.append(bool(predicted & truth))
        print(f"Первые {k} предсказаний")
        print("Средняя полнота по microcat:", round(np.mean(recalls) * 100, 2), "%")
        print("Запросов хотя бы с одним правильным microcat:", round(np.mean(hits) * 100, 2), "%")



# Для microcat использую лёгкую модель с учителем: TF-IDF-признаки и LinearSVC.
# Здесь важнее быстрый и достаточно точный список первых предсказаний для поиска кандидатов, а не отдельная тяжёлая нейросетевая модель.
def main():
    print("Загружаем обучающую выборку...")
    df = pd.read_parquet(TRAIN_PATH, columns=["search_query", "item_microcat_id"])
    # Обучающую и валидационную части делю группами по search_query. Так один и тот же текст запроса
    # не попадает в обе части и оценка классификатора microcat получается менее завышенной.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["search_query"]))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    train_pairs = train_df[["search_query", "item_microcat_id"]].drop_duplicates().reset_index(drop=True)
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_pairs["search_query"])
    y_train = train_pairs["item_microcat_id"].to_numpy()
    print("Обучающих пар:", len(train_pairs))
    print("Размер матрицы признаков:", X_train.shape)
    classifier = LinearSVC(C=2.0)
    classifier.fit(X_train, y_train)
    evaluate_top_k(classifier, vectorizer, val_df)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    print("Сохранено:", VECTORIZER_PATH)
    print("Сохранено:", CLASSIFIER_PATH)
if __name__ == "__main__":
    main()
