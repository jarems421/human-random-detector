import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from features import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
ANALYTICS_PATH = PROJECT_ROOT / "analytics.csv"
REPORT_PATH = PROJECT_ROOT / "real_data_evaluation.json"
SUPABASE_TABLE = "analytics"

REQUIRED_COLUMNS = {"sequence", "actual_label"}
LABEL_MAP = {"Random": 0, "Human": 1}
TARGET_NAMES = ["random", "human"]


def validate_required_columns(df):
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing:
        raise ValueError(f"Missing required analytics columns: {', '.join(missing)}")


def prepare_labeled_rows(df):
    validate_required_columns(df)

    sequences = []
    labels = []
    skipped = {
        "short_sequence": 0,
        "non_binary_sequence": 0,
        "invalid_label": 0,
    }

    for _, row in df.iterrows():
        label = row["actual_label"]
        sequence = row["sequence"]

        if not isinstance(label, str) or label.strip() not in LABEL_MAP:
            skipped["invalid_label"] += 1
            continue

        if not isinstance(sequence, str):
            skipped["non_binary_sequence"] += 1
            continue

        sequence = sequence.strip()

        if len(sequence) < 10:
            skipped["short_sequence"] += 1
            continue

        if not all(char in "01" for char in sequence):
            skipped["non_binary_sequence"] += 1
            continue

        sequences.append(sequence)
        labels.append(LABEL_MAP[label.strip()])

    return sequences, np.array(labels), skipped


def build_evaluation(y_true, y_pred, y_prob_human, valid_rows, skipped):
    has_both_classes = len(set(y_true)) == 2
    roc_auc = roc_auc_score(y_true, y_prob_human) if has_both_classes else None

    return {
        "valid_rows": int(valid_rows),
        "skipped_rows": int(sum(skipped.values())),
        "skipped": skipped,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=TARGET_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_dataframe(df, model, scaler):
    sequences, y_true, skipped = prepare_labeled_rows(df)

    if len(sequences) == 0:
        raise ValueError("No valid labeled rows found. Collect labeled samples in the app first.")

    features = np.array([extract_features(sequence) for sequence in sequences])
    scaled_features = scaler.transform(features)
    y_pred = model.predict(scaled_features)
    y_prob_human = model.predict_proba(scaled_features)[:, 1]

    return build_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_prob_human=y_prob_human,
        valid_rows=len(sequences),
        skipped=skipped,
    )


def print_evaluation(evaluation):
    print("Real-data evaluation complete")
    print(f"Valid rows: {evaluation['valid_rows']}")
    print(f"Skipped rows: {evaluation['skipped_rows']}")
    print(f"Skipped detail: {evaluation['skipped']}")
    print(f"Accuracy: {evaluation['accuracy']:.3f}")

    if evaluation["roc_auc"] is None:
        print("ROC AUC: skipped (both classes are required)")
    else:
        print(f"ROC AUC: {evaluation['roc_auc']:.3f}")

    print("\nConfusion matrix:")
    print(np.array(evaluation["confusion_matrix"]))
    print("\nClassification report:")
    print(pd.DataFrame(evaluation["classification_report"]).transpose())


def get_supabase_config():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        return None

    return {
        "url": url.rstrip("/"),
        "key": key,
    }


def get_supabase_headers(config):
    return {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
    }


def load_supabase_dataframe(config):
    endpoint = f"{config['url']}/rest/v1/{SUPABASE_TABLE}"
    params = {
        "select": "sequence,actual_label,session_id,batch_id,batch_position",
        "order": "created_at.asc",
    }
    response = requests.get(
        endpoint,
        headers=get_supabase_headers(config),
        params=params,
        timeout=10,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


def load_csv_dataframe():
    return pd.read_csv(
        ANALYTICS_PATH,
        dtype={
            "sequence": "string",
            "actual_label": "string",
        },
    )


def main():
    try:
        supabase_config = get_supabase_config()

        if supabase_config:
            df = load_supabase_dataframe(supabase_config)
        elif ANALYTICS_PATH.exists():
            df = load_csv_dataframe()
        else:
            print("No analytics.csv found. Collect labeled samples in the app first.")
            return 0

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        evaluation = evaluate_dataframe(df, model, scaler)
    except ValueError as exc:
        print(exc)
        return 1
    except requests.RequestException as exc:
        print(f"Could not load Supabase analytics: {exc}")
        return 1

    REPORT_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print_evaluation(evaluation)
    print(f"Saved real-data evaluation to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
