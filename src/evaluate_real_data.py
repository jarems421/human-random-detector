import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from calibration import build_calibration_report
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


def clean_sequence(sequence):
    return "".join(str(sequence).split())


def prepare_labeled_dataframe(df):
    validate_required_columns(df)

    rows = []
    seen_sequences = set()
    skipped = {
        "short_sequence": 0,
        "non_binary_sequence": 0,
        "invalid_label": 0,
        "duplicate_sequence": 0,
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

        sequence = clean_sequence(sequence)

        if len(sequence) < 10:
            skipped["short_sequence"] += 1
            continue

        if not all(char in "01" for char in sequence):
            skipped["non_binary_sequence"] += 1
            continue

        if sequence in seen_sequences:
            skipped["duplicate_sequence"] += 1
            continue

        seen_sequences.add(sequence)
        label = label.strip()
        rows.append(
            {
                "sequence": sequence,
                "actual_label": label,
                "label": LABEL_MAP[label],
                "session_id": row.get("session_id"),
                "batch_id": row.get("batch_id"),
                "batch_position": row.get("batch_position"),
            }
        )

    return pd.DataFrame(rows), skipped


def prepare_labeled_rows(df):
    labeled_df, skipped = prepare_labeled_dataframe(df)

    if labeled_df.empty:
        return [], np.array([], dtype=int), skipped

    return (
        labeled_df["sequence"].tolist(),
        labeled_df["label"].to_numpy(dtype=int),
        skipped,
    )


def build_evaluation(y_true, y_pred, y_prob_human, valid_rows, skipped, group_summary=None):
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
        "calibration": build_calibration_report(y_true, y_prob_human),
        "group_summary": group_summary or {},
    }


def evaluate_dataframe(df, model, scaler):
    labeled_df, skipped = prepare_labeled_dataframe(df)

    if labeled_df.empty:
        raise ValueError("No valid labeled rows found. Collect labeled samples in the app first.")

    sequences = labeled_df["sequence"].tolist()
    y_true = labeled_df["label"].to_numpy(dtype=int)
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
        group_summary=summarize_groups(labeled_df),
    )


def summarize_groups(df):
    return {
        "rows_with_session_id": int(df["session_id"].notna().sum()) if "session_id" in df else 0,
        "session_count": int(df["session_id"].dropna().nunique()) if "session_id" in df else 0,
        "rows_with_batch_id": int(df["batch_id"].notna().sum()) if "batch_id" in df else 0,
        "batch_count": int(df["batch_id"].dropna().nunique()) if "batch_id" in df else 0,
    }


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
    print(f"\nBrier score: {evaluation['calibration']['brier_score']:.3f}")
    print(f"Calibration: {evaluation['calibration']['summary']}")


def get_supabase_config():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

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
            "session_id": "string",
            "batch_id": "string",
        },
    )


def main():
    try:
        supabase_config = get_supabase_config()

        if supabase_config:
            df = load_supabase_dataframe(supabase_config)
        elif ANALYTICS_PATH.exists():
            df = load_csv_dataframe()
        elif os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_KEY"):
            print(
                "Raw Supabase evaluation requires SUPABASE_SERVICE_ROLE_KEY. "
                "Set it locally or export analytics.csv."
            )
            return 1
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
