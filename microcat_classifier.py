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
    """Смешиваем word и char n-grams: на коротких и кривоватых сервисных запросах вместе они работают стабильнее."""
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
    """Смотрим top-k coverage, чтобы понимать, сколько microcat имеет смысл брать в retrieval."""
    val_queries = (
        val_df.groupby("search_query")["item_microcat_id"].agg(set).reset_index(name="true_microcats")
    )
    X_val = vectorizer.transform(val_queries["search_query"])
    scores = classifier.decision_function(X_val)
    classes = classifier.classes_
    print("Validation queries:", len(val_queries))
    for k in [1, 2, 3, 5, 10]:
        top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
        recalls, hits = [], []
        for i, row in val_queries.iterrows():
            predicted = set(classes[top_indices[i]])
            truth = row["true_microcats"]
            recalls.append(len(predicted & truth) / len(truth))
            hits.append(bool(predicted & truth))
        print(f"TOP-{k}")
        print("Mean microcat recall:", round(np.mean(recalls) * 100, 2), "%")
        print("Queries with >=1 correct microcat:", round(np.mean(hits) * 100, 2), "%")



# Обучение небольшое: TF-IDF признаки + LinearSVC, без тяжёлой нейросети.
def main():
    print("Loading train...")
    df = pd.read_parquet(TRAIN_PATH, columns=["search_query", "item_microcat_id"])
    # Деление делаю именно по search_query, чтобы один и тот же текст не оказался одновременно в train и validation.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["search_query"]))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    train_pairs = train_df[["search_query", "item_microcat_id"]].drop_duplicates().reset_index(drop=True)
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_pairs["search_query"])
    y_train = train_pairs["item_microcat_id"].to_numpy()
    print("Train pairs:", len(train_pairs))
    print("Feature matrix:", X_train.shape)
    classifier = LinearSVC(C=2.0)
    classifier.fit(X_train, y_train)
    evaluate_top_k(classifier, vectorizer, val_df)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    print("Saved:", VECTORIZER_PATH)
    print("Saved:", CLASSIFIER_PATH)
if __name__ == "__main__":
    main()
