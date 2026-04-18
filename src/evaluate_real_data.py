import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from calibration import build_calibration_report
from features import extract_features
from real_data import (
    ANALYTICS_PATH,
    LABEL_MAP,
    TARGET_NAMES,
    get_supabase_config,
    load_csv_dataframe,
    load_supabase_dataframe,
    missing_private_supabase_key,
    prepare_labeled_dataframe,
    prepare_labeled_rows,
    summarize_groups,
    validate_required_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
REPORT_PATH = PROJECT_ROOT / "real_data_evaluation.json"


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


def main():
    try:
        supabase_config = get_supabase_config()

        if supabase_config:
            df = load_supabase_dataframe(supabase_config)
        elif ANALYTICS_PATH.exists():
            df = load_csv_dataframe()
        elif missing_private_supabase_key():
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
