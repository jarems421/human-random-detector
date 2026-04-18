import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from calibration import build_calibration_report
from features import extract_features
from generate_data import create_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
REPORT_PATH = PROJECT_ROOT / "evaluation_report.json"


def build_feature_matrix(sequences):
    return np.array([extract_features(sequence) for sequence in sequences])


def train_and_save_model(n_samples=1000, length=50, test_size=0.2, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    sequences, labels = create_dataset(n=n_samples, length=length)
    labels = np.array(labels)

    x = build_feature_matrix(sequences)
    y = labels

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = GaussianNB()
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    y_prob_human = model.predict_proba(x_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob_human)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["random", "human"],
        output_dict=True,
    )

    evaluation = {
        "samples": int(len(y)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "sequence_length": int(length),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "calibration": build_calibration_report(y_test, y_prob_human),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    REPORT_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Samples: {len(y)} ({len(y_train)} train, {len(y_test)} test)")
    print(f"Sequence length: {length}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["random", "human"],
        )
    )
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved evaluation report to: {REPORT_PATH}")

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "report_path": REPORT_PATH,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the human/random detector.")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per class")
    parser.add_argument("--length", type=int, default=50, help="Sequence length")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save_model(
        n_samples=args.samples,
        length=args.length,
        test_size=args.test_size,
        random_state=args.seed,
    )
