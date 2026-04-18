import json
from pathlib import Path

import pandas as pd

from features import FEATURE_NAMES, extract_feature_dict
from generate_data import create_dataset
from real_data import load_real_dataframe, prepare_labeled_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "real_synthetic_comparison.json"


def summarize_features(sequences):
    if not sequences:
        return {name: None for name in FEATURE_NAMES}

    feature_df = pd.DataFrame([extract_feature_dict(sequence) for sequence in sequences])
    return {
        feature: float(feature_df[feature].mean())
        for feature in FEATURE_NAMES
    }


def build_comparison(real_df, synthetic_per_class=1000, length=50):
    real_labeled_df, skipped = prepare_labeled_dataframe(real_df)

    if real_labeled_df.empty:
        raise ValueError("No valid labeled real rows found.")

    synthetic_sequences, synthetic_labels = create_dataset(n=synthetic_per_class, length=length)
    synthetic_df = pd.DataFrame(
        {
            "sequence": synthetic_sequences,
            "actual_label": ["Random" if label == 0 else "Human" for label in synthetic_labels],
        }
    )

    return {
        "real_rows": int(len(real_labeled_df)),
        "synthetic_rows": int(len(synthetic_df)),
        "skipped_real_rows": int(sum(skipped.values())),
        "skipped": skipped,
        "real": summarize_by_label(real_labeled_df),
        "synthetic": summarize_by_label(synthetic_df),
    }


def summarize_by_label(df):
    summary = {}

    for label, group in df.groupby("actual_label"):
        summary[label] = {
            "count": int(len(group)),
            "feature_means": summarize_features(group["sequence"].tolist()),
        }

    return summary


def main():
    try:
        real_df = load_real_dataframe()
        comparison = build_comparison(real_df)
    except ValueError as exc:
        print(exc)
        return 1

    REPORT_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("Real-vs-synthetic feature comparison complete")
    print(f"Real rows: {comparison['real_rows']}")
    print(f"Synthetic rows: {comparison['synthetic_rows']}")
    print(f"Saved comparison to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
