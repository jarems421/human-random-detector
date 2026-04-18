import numpy as np
import pandas as pd
import pytest

from evaluate_real_data import build_evaluation, get_supabase_config, prepare_labeled_rows


def test_prepare_labeled_rows_maps_valid_labels():
    df = pd.DataFrame(
        [
            {"sequence": "0101010101", "actual_label": "Human"},
            {"sequence": "0011001100", "actual_label": "Random"},
        ]
    )

    sequences, labels, skipped = prepare_labeled_rows(df)

    assert sequences == ["0101010101", "0011001100"]
    assert labels.tolist() == [1, 0]
    assert sum(skipped.values()) == 0


def test_prepare_labeled_rows_skips_invalid_sequences():
    df = pd.DataFrame(
        [
            {"sequence": "0101", "actual_label": "Human"},
            {"sequence": "0101010102", "actual_label": "Random"},
        ]
    )

    sequences, labels, skipped = prepare_labeled_rows(df)

    assert sequences == []
    assert labels.tolist() == []
    assert skipped["short_sequence"] == 1
    assert skipped["non_binary_sequence"] == 1


def test_prepare_labeled_rows_skips_invalid_labels():
    df = pd.DataFrame(
        [
            {"sequence": "0101010101", "actual_label": "Unknown"},
            {"sequence": "0011001100", "actual_label": None},
        ]
    )

    sequences, labels, skipped = prepare_labeled_rows(df)

    assert sequences == []
    assert labels.tolist() == []
    assert skipped["invalid_label"] == 2


def test_prepare_labeled_rows_requires_columns():
    df = pd.DataFrame([{"sequence": "0101010101"}])

    with pytest.raises(ValueError, match="actual_label"):
        prepare_labeled_rows(df)


def test_build_evaluation_skips_roc_auc_for_one_class():
    evaluation = build_evaluation(
        y_true=np.array([1, 1]),
        y_pred=np.array([1, 0]),
        y_prob_human=np.array([0.9, 0.4]),
        valid_rows=2,
        skipped={"short_sequence": 0, "non_binary_sequence": 0, "invalid_label": 0},
    )

    assert evaluation["roc_auc"] is None
    assert evaluation["accuracy"] == pytest.approx(0.5)
    assert evaluation["confusion_matrix"] == [[0, 0], [1, 1]]


def test_prepare_labeled_rows_deduplicates_sequences():
    df = pd.DataFrame(
        [
            {"sequence": "0101010101", "actual_label": "Human"},
            {"sequence": "0101010101", "actual_label": "Human"},
            {"sequence": " 01010 10101 ", "actual_label": "Human"},
        ]
    )

    sequences, labels, skipped = prepare_labeled_rows(df)

    assert sequences == ["0101010101"]
    assert labels.tolist() == [1]
    assert skipped["duplicate_sequence"] == 2


def test_supabase_config_requires_service_role_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "anon-key")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    assert get_supabase_config() is None


def test_supabase_config_uses_service_role_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-key")

    config = get_supabase_config()

    assert config == {
        "url": "https://example.supabase.co",
        "key": "service-role-key",
    }
