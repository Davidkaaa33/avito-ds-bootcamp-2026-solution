from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(__file__).resolve().parent / "data"

print("Загружаем train...")

df = pd.read_parquet(
    DATA_DIR / "train.parquet",
    columns=[
        "search_query",
        "item_microcat_id",
    ]
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        df,
        groups=df["search_query"]
    )
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

train_pairs = (
    train_df[
        ["search_query", "item_microcat_id"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("Train pairs:", len(train_pairs))

print("Строим TF-IDF признаки...")

vectorizer = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True
        )
    ),

    (
        "char",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=120_000,
            sublinear_tf=True
        )
    ),
])

X_train = vectorizer.fit_transform(
    train_pairs["search_query"]
)

y_train = (
    train_pairs["item_microcat_id"]
    .to_numpy()
)

print("Размер признаков:", X_train.shape)

print("Обучаем microcat classifier...")

classifier = LinearSVC(
    C=2.0
)

classifier.fit(
    X_train,
    y_train
)

print("Classifier обучен.")

val_queries = (
    val_df.groupby("search_query")[
        "item_microcat_id"
    ]
    .agg(set)
    .reset_index(name="true_microcats")
)

print( "Validation queries:",len(val_queries))

X_val = vectorizer.transform(
    val_queries["search_query"]
)

scores = classifier.decision_function(
    X_val
)

classes = classifier.classes_

for k in [1, 2, 3, 5, 10]:

    top_indices = np.argpartition(
        scores,
        -k,
        axis=1
    )[:, -k:]

    recalls = []
    hit = []

    for i in range(len(val_queries)):

        predicted = set(
            classes[
                top_indices[i]
            ]
        )

        truth = (
            val_queries.iloc[i][
                "true_microcats"
            ]
        )

        recalls.append(
            len(predicted & truth)
            / len(truth)
        )

        hit.append(
            len(predicted & truth) > 0
        )

    print()
    print(f"TOP-{k}")

    print( "Mean microcat recall:",round(np.mean(recalls) * 100,2), "%")

    print( "Queries with >=1 correct microcat:",round(np.mean(hit) * 100,2), "%")

joblib.dump(
    vectorizer,
    DATA_DIR / "microcat_vectorizer.joblib"
)

joblib.dump(
    classifier,
    DATA_DIR / "microcat_classifier.joblib"
)

print("\nМодель сохранена.")
