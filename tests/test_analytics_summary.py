import pandas as pd
import pytest

from analytics_summary import build_public_summary, label_count_frame, summary_from_supabase_row


def test_build_public_summary_uses_aggregate_metrics_only():
    df = pd.DataFrame(
        [
            {
                "sequence": "0101010101",
                "actual_label": "Human",
                "model_prediction": "Human",
                "user_guess": "Human",
                "p_human": 0.9,
            },
            {
                "sequence": "0011001100",
                "actual_label": "Random",
                "model_prediction": "Human",
                "user_guess": None,
                "p_human": 0.6,
            },
            {
                "sequence": "1110001110",
                "actual_label": "Random",
                "model_prediction": "Random",
                "user_guess": "Human",
                "p_human": 0.2,
            },
        ]
    )

    summary = build_public_summary(df)

    assert summary["total_rows"] == 3
    assert summary["human_rows"] == 1
    assert summary["random_rows"] == 2
    assert summary["model_accuracy"] == pytest.approx(2 / 3)
    assert summary["human_precision"] == pytest.approx(0.5)
    assert summary["human_recall"] == pytest.approx(1.0)
    assert summary["random_recall"] == pytest.approx(0.5)
    assert "sequence" not in summary


def test_label_count_frame_has_human_and_random_rows():
    summary = {"human_rows": 4, "random_rows": 6}
    frame = label_count_frame(summary)

    assert frame.to_dict("records") == [
        {"label": "Human", "rows": 4},
        {"label": "Random", "rows": 6},
    ]


def test_summary_from_supabase_row_casts_numbers():
    summary = summary_from_supabase_row(
        {
            "total_rows": 10,
            "human_rows": 4,
            "random_rows": 6,
            "model_accuracy": "0.8",
            "guessed_rows": 2,
        }
    )

    assert summary["total_rows"] == 10
    assert summary["model_accuracy"] == pytest.approx(0.8)
