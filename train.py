import os
import pandas as pd
import joblib
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ARTIFACTS_DIR = "artifacts"

def load_data():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "patrickb1912/ipl-complete-dataset-20082020",
        "matches.csv"
    )
    required = ["season","venue","team1","team2","toss_winner","toss_decision","winner"]
    df = df[required].dropna().copy()
    df["toss_decision"] = df["toss_decision"].str.lower().replace({"bat":"bat","field":"field"})
    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)
    return df

def build_pipeline(cat_features):
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = load_data()
    cat_features = ["season","venue","team1","team2","toss_winner","toss_decision"]
    X = df[cat_features]
    y = df["team1_win"]
    pipe = build_pipeline(cat_features)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:,1]
    auc = roc_auc_score(y, proba)
    print(f"In-sample ROC AUC: {auc:.3f}")
    joblib.dump(pipe, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.txt"), "w") as f:
        f.write(",".join(cat_features))
    print("Artifacts saved in", ARTIFACTS_DIR)

if __name__ == "__main__":
    main()
