import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from calibration import build_calibration_report
from features import extract_features
from generate_data import create_dataset
from real_data import (
    TARGET_NAMES,
    class_counts,
    load_real_dataframe,
    missing_private_supabase_key,
    prepare_labeled_dataframe,
    select_group_ids,
    summarize_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
REPORT_PATH = PROJECT_ROOT / "real_core_training_report.json"
REAL_EVALUATION_PATH = PROJECT_ROOT / "real_data_evaluation.json"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CANDIDATE_MODEL_PATH = EXPERIMENTS_DIR / "real_core_candidate_model.pkl"
CANDIDATE_SCALER_PATH = EXPERIMENTS_DIR / "real_core_candidate_scaler.pkl"

MIN_HOLDOUT_ROWS = 100
MIN_HOLDOUT_HUMAN = 30
MIN_HOLDOUT_RANDOM = 30
MACRO_F1_MARGIN = 0.005
MAX_ROC_AUC_DROP = 0.03
MAX_HUMAN_PRECISION_DROP = 0.05


def build_feature_matrix(sequences):
    return np.array([extract_features(sequence) for sequence in sequences])


def split_real_dataframe(df, test_size=0.25, random_state=42):
    df = df.reset_index(drop=True).copy()
    mode, group_ids = select_group_ids(df)
    warning = None

    if mode != "row_level_fallback" and group_ids.nunique() >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(df, groups=group_ids))
    else:
        mode = "row_level_fallback"
        warning = "lower confidence evaluation due to missing grouping metadata"
        group_ids = pd.Series(
            [f"row_{index}" for index in df.index],
            index=df.index,
            dtype="string",
        )
        stratify = df["label"] if df["label"].nunique() == 2 else None
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_groups = sorted(set(group_ids.iloc[train_idx].astype(str)))
    test_groups = sorted(set(group_ids.iloc[test_idx].astype(str)))

    return {
        "train_df": train_df,
        "test_df": test_df,
        "report": {
            "random_seed": int(random_state),
            "test_size": float(test_size),
            "group_split_mode": mode,
            "warning": warning,
            "train_group_ids": train_groups,
            "test_group_ids": test_groups,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_class_counts": class_counts(train_df["label"]),
            "test_class_counts": class_counts(test_df["label"]),
        },
    }


def generate_synthetic_support(real_train_rows, ratio=1.0, length=50, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    max_synthetic_rows = int(real_train_rows * ratio)
    samples_per_class = max_synthetic_rows // 2

    if samples_per_class <= 0:
        return pd.DataFrame(columns=["sequence", "label", "actual_label", "source"])

    sequences, labels = create_dataset(n=samples_per_class, length=length)
    return pd.DataFrame(
        {
            "sequence": sequences,
            "label": labels,
            "actual_label": ["Random" if label == 0 else "Human" for label in labels],
            "source": "synthetic",
        }
    )


def build_training_data(
    real_train_df,
    synthetic_ratio=1.0,
    sequence_length=50,
    random_state=42,
    real_weight=3.0,
    synthetic_weight=1.0,
):
    real_df = real_train_df[["sequence", "label", "actual_label"]].copy()
    real_df["source"] = "real"
    synthetic_df = generate_synthetic_support(
        real_train_rows=len(real_df),
        ratio=synthetic_ratio,
        length=sequence_length,
        random_state=random_state,
    )
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    sample_weights = np.where(
        combined_df["source"].to_numpy() == "real",
        real_weight,
        synthetic_weight,
    )

    return combined_df, sample_weights


def train_candidate_model(training_df, sample_weights):
    x = build_feature_matrix(training_df["sequence"].tolist())
    y = training_df["label"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = GaussianNB()
    model.fit(x_scaled, y, sample_weight=sample_weights)

    return model, scaler


def evaluate_model(model, scaler, df):
    y_true = df["label"].to_numpy(dtype=int)
    x = build_feature_matrix(df["sequence"].tolist())
    x_scaled = scaler.transform(x)
    y_pred = model.predict(x_scaled)
    y_prob_human = model.predict_proba(x_scaled)[:, 1]
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_true, y_prob_human) if len(set(y_true)) == 2 else None

    return {
        "valid_rows": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": report,
        "calibration": build_calibration_report(y_true, y_prob_human),
    }


def holdout_evidence_check(
    test_class_counts,
    min_rows=MIN_HOLDOUT_ROWS,
    min_human=MIN_HOLDOUT_HUMAN,
    min_random=MIN_HOLDOUT_RANDOM,
):
    total = test_class_counts["Human"] + test_class_counts["Random"]
    failures = []

    if total < min_rows:
        failures.append(f"holdout rows {total} < {min_rows}")

    if test_class_counts["Human"] < min_human:
        failures.append(f"human holdout rows {test_class_counts['Human']} < {min_human}")

    if test_class_counts["Random"] < min_random:
        failures.append(f"random holdout rows {test_class_counts['Random']} < {min_random}")

    return {
        "sufficient": not failures,
        "failures": failures,
        "minimums": {
            "rows": int(min_rows),
            "human_rows": int(min_human),
            "random_rows": int(min_random),
        },
    }


def get_metric(evaluation, label, metric):
    return evaluation["classification_report"][label][metric]


def promotion_decision(
    baseline,
    candidate,
    holdout_check,
    macro_f1_margin=MACRO_F1_MARGIN,
    max_roc_auc_drop=MAX_ROC_AUC_DROP,
    max_human_precision_drop=MAX_HUMAN_PRECISION_DROP,
):
    if not holdout_check["sufficient"]:
        return {
            "promoted": False,
            "promotion_decision": "insufficient_evidence",
            "reasons": holdout_check["failures"],
        }

    baseline_human_recall = get_metric(baseline, "human", "recall")
    candidate_human_recall = get_metric(candidate, "human", "recall")
    baseline_human_precision = get_metric(baseline, "human", "precision")
    candidate_human_precision = get_metric(candidate, "human", "precision")
    baseline_macro_f1 = baseline["classification_report"]["macro avg"]["f1-score"]
    candidate_macro_f1 = candidate["classification_report"]["macro avg"]["f1-score"]
    baseline_roc_auc = baseline["roc_auc"]
    candidate_roc_auc = candidate["roc_auc"]

    checks = {
        "human_recall_stable": candidate_human_recall >= baseline_human_recall,
        "macro_f1_improves": candidate_macro_f1 >= baseline_macro_f1 + macro_f1_margin,
        "roc_auc_stable": (
            baseline_roc_auc is not None
            and candidate_roc_auc is not None
            and candidate_roc_auc >= baseline_roc_auc - max_roc_auc_drop
        ),
        "human_precision_stable": (
            candidate_human_precision >= baseline_human_precision - max_human_precision_drop
        ),
    }

    macro_tie_with_recall_gain = (
        abs(candidate_macro_f1 - baseline_macro_f1) < 1e-12
        and candidate_human_recall > baseline_human_recall
    )

    if macro_tie_with_recall_gain:
        checks["macro_f1_improves"] = True

    promoted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]

    return {
        "promoted": promoted,
        "promotion_decision": "promoted" if promoted else "not_promoted",
        "checks": checks,
        "reasons": failed,
        "thresholds": {
            "macro_f1_margin": float(macro_f1_margin),
            "max_roc_auc_drop": float(max_roc_auc_drop),
            "max_human_precision_drop": float(max_human_precision_drop),
            "no_promotion_on_ties_unless_recall_improves": True,
        },
        "metric_deltas": {
            "human_recall": float(candidate_human_recall - baseline_human_recall),
            "human_precision": float(candidate_human_precision - baseline_human_precision),
            "macro_f1": float(candidate_macro_f1 - baseline_macro_f1),
            "roc_auc": (
                None
                if baseline_roc_auc is None or candidate_roc_auc is None
                else float(candidate_roc_auc - baseline_roc_auc)
            ),
        },
    }


def weight_accounting(training_df, sample_weights, real_weight, synthetic_weight):
    rows = []
    weighted = {}

    for source in ["real", "synthetic"]:
        source_mask = training_df["source"] == source
        source_df = training_df[source_mask]
        source_weights = sample_weights[source_mask.to_numpy()]
        weighted[source] = {
            "rows": int(len(source_df)),
            "sample_weight": float(real_weight if source == "real" else synthetic_weight),
            "class_counts": class_counts(source_df["label"]),
            "effective_weight_by_class": {},
        }

        for label_name, label_value in [("Random", 0), ("Human", 1)]:
            mask = source_df["label"].to_numpy(dtype=int) == label_value
            weighted[source]["effective_weight_by_class"][label_name] = float(
                source_weights[mask].sum()
            )
            rows.append(
                {
                    "source": source,
                    "label": label_name,
                    "rows": int(mask.sum()),
                    "effective_weight": float(source_weights[mask].sum()),
                }
            )

    return {
        "real_sample_weight": float(real_weight),
        "synthetic_sample_weight": float(synthetic_weight),
        "by_source": weighted,
        "rows": rows,
    }


def save_candidate_artifacts(model, scaler):
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, CANDIDATE_MODEL_PATH)
    joblib.dump(scaler, CANDIDATE_SCALER_PATH)


def write_report(report):
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_real_core_training(
    test_size=0.25,
    random_state=42,
    synthetic_to_real_ratio=1.0,
    sequence_length=50,
    real_weight=3.0,
    synthetic_weight=1.0,
):
    random.seed(random_state)
    np.random.seed(random_state)

    raw_df = load_real_dataframe()
    real_df, skipped = prepare_labeled_dataframe(raw_df)

    if real_df.empty:
        raise ValueError("No valid labeled real rows found.")

    split = split_real_dataframe(
        real_df,
        test_size=test_size,
        random_state=random_state,
    )
    training_df, sample_weights = build_training_data(
        split["train_df"],
        synthetic_ratio=synthetic_to_real_ratio,
        sequence_length=sequence_length,
        random_state=random_state,
        real_weight=real_weight,
        synthetic_weight=synthetic_weight,
    )
    candidate_model, candidate_scaler = train_candidate_model(training_df, sample_weights)
    baseline_model = joblib.load(MODEL_PATH)
    baseline_scaler = joblib.load(SCALER_PATH)
    baseline_eval = evaluate_model(baseline_model, baseline_scaler, split["test_df"])
    candidate_eval = evaluate_model(candidate_model, candidate_scaler, split["test_df"])
    holdout_check = holdout_evidence_check(split["report"]["test_class_counts"])
    decision = promotion_decision(baseline_eval, candidate_eval, holdout_check)

    report = {
        "config": {
            "random_state": int(random_state),
            "test_size": float(test_size),
            "synthetic_to_real_ratio": float(synthetic_to_real_ratio),
            "sequence_length": int(sequence_length),
            "real_weight": float(real_weight),
            "synthetic_weight": float(synthetic_weight),
        },
        "real_data": {
            "raw_rows": int(len(raw_df)),
            "valid_rows": int(len(real_df)),
            "skipped_rows": int(sum(skipped.values())),
            "skipped": skipped,
            "group_summary": summarize_groups(real_df),
        },
        "split": split["report"],
        "training": {
            "real_training_rows": int(len(split["train_df"])),
            "synthetic_support_rows": int((training_df["source"] == "synthetic").sum()),
            "synthetic_support_cap_ratio": float(synthetic_to_real_ratio),
            "weight_accounting": weight_accounting(
                training_df,
                sample_weights,
                real_weight=real_weight,
                synthetic_weight=synthetic_weight,
            ),
        },
        "baseline_evaluation": baseline_eval,
        "candidate_evaluation": candidate_eval,
        "holdout_check": holdout_check,
        "promotion": decision,
    }

    if decision["promoted"]:
        joblib.dump(candidate_model, MODEL_PATH)
        joblib.dump(candidate_scaler, SCALER_PATH)
        REAL_EVALUATION_PATH.write_text(json.dumps(candidate_eval, indent=2), encoding="utf-8")
    else:
        save_candidate_artifacts(candidate_model, candidate_scaler)
        report["candidate_artifacts"] = {
            "model_path": str(CANDIDATE_MODEL_PATH),
            "scaler_path": str(CANDIDATE_SCALER_PATH),
        }

    write_report(report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Train a real-core hybrid candidate model.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-to-real-ratio", type=float, default=1.0)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--real-weight", type=float, default=3.0)
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        report = run_real_core_training(
            test_size=args.test_size,
            random_state=args.seed,
            synthetic_to_real_ratio=args.synthetic_to_real_ratio,
            sequence_length=args.length,
            real_weight=args.real_weight,
            synthetic_weight=args.synthetic_weight,
        )
    except ValueError as exc:
        print(exc)

        if missing_private_supabase_key():
            print("Set SUPABASE_SERVICE_ROLE_KEY for private Supabase training.")

        return 1

    print("Real-core training experiment complete")
    print(f"Promotion decision: {report['promotion']['promotion_decision']}")
    print(f"Baseline accuracy: {report['baseline_evaluation']['accuracy']:.3f}")
    print(f"Candidate accuracy: {report['candidate_evaluation']['accuracy']:.3f}")
    print(f"Saved report to: {REPORT_PATH}")

    if not report["promotion"]["promoted"]:
        print(f"Saved candidate model to: {CANDIDATE_MODEL_PATH}")
        print(f"Saved candidate scaler to: {CANDIDATE_SCALER_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
