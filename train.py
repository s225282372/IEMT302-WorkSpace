import argparse
import json
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.config import LABEL_COLUMN
from src.model import build_model


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    y = (df[LABEL_COLUMN].astype(str).str.upper() == "Y").astype(int).values
    X = df.drop(columns=[LABEL_COLUMN])
    return X, y


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy@{:.2f}".format(threshold): float(accuracy_score(y_true, y_pred)),
        "f1@{:.2f}".format(threshold): float(f1_score(y_true, y_pred)),
        "confusion_matrix@{:.2f}".format(threshold): confusion_matrix(y_true, y_pred).tolist(),
        "classification_report@{:.2f}".format(threshold): classification_report(y_true, y_pred, output_dict=True),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train course completion model")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--no_calibrate", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data_path)
    X, y = split_features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = build_model(random_state=args.random_state, calibrate=(not args.no_calibrate))

    # Handle potential imbalance with balanced sample weights
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model.fit(X_train, y_train, model__sample_weight=sample_weight)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_prob, threshold=args.threshold)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.joblib")
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()