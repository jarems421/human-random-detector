from pathlib import Path

import joblib
import pandas as pd
import pytest

import train_real_core_model
from train_real_core_model import (
    build_training_data,
    holdout_evidence_check,
    promotion_decision,
    save_candidate_artifacts,
    split_real_dataframe,
    weight_accounting,
)


def make_labeled_rows(rows=40, with_batch=True):
    data = []

    for index in range(rows):
        label = index % 2
        sequence = format(index, "010b")
        data.append(
            {
                "sequence": sequence,
                "actual_label": "Human" if label else "Random",
                "label": label,
                "batch_id": f"batch-{index // 4}" if with_batch else None,
                "session_id": f"session-{index // 8}" if with_batch else None,
                "batch_position": index % 4 + 1,
            }
        )

    return pd.DataFrame(data)


def test_group_split_keeps_group_ids_separate():
    df = make_labeled_rows(rows=40, with_batch=True)
    split = split_real_dataframe(df, random_state=42, test_size=0.25)

    train_groups = set(split["report"]["train_group_ids"])
    test_groups = set(split["report"]["test_group_ids"])

    assert split["report"]["group_split_mode"] == "batch_id"
    assert train_groups.isdisjoint(test_groups)
    assert split["report"]["train_class_counts"]["Human"] > 0
    assert split["report"]["test_class_counts"]["Random"] > 0


def test_row_level_fallback_sets_warning_and_mode():
    df = make_labeled_rows(rows=20, with_batch=False)
    split = split_real_dataframe(df, random_state=42, test_size=0.25)

    assert split["report"]["group_split_mode"] == "row_level_fallback"
    assert split["report"]["warning"] == "lower confidence evaluation due to missing grouping metadata"


def test_synthetic_support_is_capped_by_ratio():
    real_train_df = make_labeled_rows(rows=20, with_batch=True)
    training_df, _ = build_training_data(
        real_train_df,
        synthetic_ratio=0.5,
        random_state=42,
    )

    synthetic_rows = int((training_df["source"] == "synthetic").sum())

    assert synthetic_rows <= 10
    assert synthetic_rows % 2 == 0


def test_real_weight_is_higher_and_reported_by_class():
    real_train_df = make_labeled_rows(rows=10, with_batch=True)
    training_df, weights = build_training_data(
        real_train_df,
        synthetic_ratio=0.4,
        real_weight=3.0,
        synthetic_weight=1.0,
        random_state=42,
    )
    accounting = weight_accounting(training_df, weights, real_weight=3.0, synthetic_weight=1.0)

    assert accounting["real_sample_weight"] > accounting["synthetic_sample_weight"]
    assert accounting["by_source"]["real"]["effective_weight_by_class"]["Human"] > 0
    assert accounting["by_source"]["synthetic"]["effective_weight_by_class"]["Random"] > 0


def test_holdout_gate_blocks_small_or_imbalanced_evidence():
    check = holdout_evidence_check({"Human": 10, "Random": 90})

    assert not check["sufficient"]
    assert any("human holdout rows" in failure for failure in check["failures"])


def test_promotion_rule_blocks_ties_without_recall_gain():
    baseline = evaluation(human_recall=0.8, human_precision=0.8, macro_f1=0.8, roc_auc=0.9)
    candidate = evaluation(human_recall=0.8, human_precision=0.8, macro_f1=0.8, roc_auc=0.9)
    holdout_check = {"sufficient": True, "failures": []}

    decision = promotion_decision(baseline, candidate, holdout_check)

    assert not decision["promoted"]
    assert "macro_f1_improves" in decision["reasons"]


def test_promotion_rule_promotes_when_gates_pass():
    baseline = evaluation(human_recall=0.8, human_precision=0.8, macro_f1=0.8, roc_auc=0.9)
    candidate = evaluation(human_recall=0.82, human_precision=0.79, macro_f1=0.81, roc_auc=0.88)
    holdout_check = {"sufficient": True, "failures": []}

    decision = promotion_decision(baseline, candidate, holdout_check)

    assert decision["promoted"]


def test_candidate_artifacts_are_saved_separately(tmp_path, monkeypatch):
    model_path = tmp_path / "candidate_model.pkl"
    scaler_path = tmp_path / "candidate_scaler.pkl"
    monkeypatch.setattr(train_real_core_model, "EXPERIMENTS_DIR", Path(tmp_path))
    monkeypatch.setattr(train_real_core_model, "CANDIDATE_MODEL_PATH", model_path)
    monkeypatch.setattr(train_real_core_model, "CANDIDATE_SCALER_PATH", scaler_path)

    save_candidate_artifacts({"model": True}, {"scaler": True})

    assert joblib.load(model_path) == {"model": True}
    assert joblib.load(scaler_path) == {"scaler": True}


def evaluation(human_recall, human_precision, macro_f1, roc_auc):
    return {
        "roc_auc": roc_auc,
        "classification_report": {
            "human": {
                "recall": human_recall,
                "precision": human_precision,
            },
            "macro avg": {
                "f1-score": macro_f1,
            },
        },
    }
