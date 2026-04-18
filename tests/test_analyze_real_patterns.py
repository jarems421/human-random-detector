import pandas as pd
import pytest

from analyze_real_patterns import (
    analyze_dataframe,
    longest_alternating_run,
    near_alternation_score,
    pattern_break_count,
    summarize_batches,
)


def test_longest_alternating_run_for_perfect_alternation():
    assert longest_alternating_run("0101010101") == 10


def test_longest_alternating_run_for_repeated_values():
    assert longest_alternating_run("0000000000") == 1


def test_near_alternation_score_for_perfect_alternation():
    assert near_alternation_score("0101010101") == pytest.approx(1.0)


def test_near_alternation_score_for_repeated_values():
    assert near_alternation_score("0000000000") == pytest.approx(0.5)


def test_pattern_break_count_for_disrupted_alternation():
    assert pattern_break_count("0101101010") > 0


def test_analyze_dataframe_groups_human_and_random_rows():
    df = pd.DataFrame(
        [
            {"sequence": "0101010101", "actual_label": "Human"},
            {"sequence": "0000000000", "actual_label": "Random"},
        ]
    )

    analysis = analyze_dataframe(df)

    assert analysis["valid_rows"] == 2
    assert analysis["label_summary"]["Human"]["count"] == 1
    assert analysis["label_summary"]["Random"]["count"] == 1
    assert analysis["label_summary"]["Human"]["mean_near_alternation_score"] == pytest.approx(1.0)


def test_batch_summary_ignores_rows_without_batch_id():
    df = pd.DataFrame(
        [
            {
                "sequence": "0101010101",
                "actual_label": "Human",
                "batch_id": None,
                "batch_position": None,
            }
        ]
    )

    summary = summarize_batches(df)

    assert summary["tracked_batch_count"] == 0
    assert summary["rows_with_batch_metadata"] == 0


def test_batch_start_bit_switching_uses_position_order():
    df = pd.DataFrame(
        [
            {
                "sequence": "0101010101",
                "actual_label": "Human",
                "batch_id": "batch-1",
                "batch_position": 1,
            },
            {
                "sequence": "1010101010",
                "actual_label": "Human",
                "batch_id": "batch-1",
                "batch_position": 2,
            },
            {
                "sequence": "1011001100",
                "actual_label": "Human",
                "batch_id": "batch-1",
                "batch_position": 3,
            },
        ]
    )

    summary = summarize_batches(df)

    assert summary["tracked_batch_count"] == 1
    assert summary["human_batch_count"] == 1
    assert summary["mean_human_batch_size"] == pytest.approx(3.0)
    assert summary["human_batches_with_both_start_bits_pct"] == pytest.approx(1.0)
    assert summary["human_adjacent_start_bit_switch_pct"] == pytest.approx(0.5)


def test_batch_previous_end_to_next_start_switching():
    df = pd.DataFrame(
        [
            {
                "sequence": "0101010101",
                "actual_label": "Human",
                "batch_id": "batch-1",
                "batch_position": 1,
            },
            {
                "sequence": "0101010100",
                "actual_label": "Human",
                "batch_id": "batch-1",
                "batch_position": 2,
            },
            {
                "sequence": "0101010101",
                "actual_label": "Human",
                "batch_id": "batch-1",
                "batch_position": 3,
            },
        ]
    )

    summary = summarize_batches(df)

    assert summary["human_previous_end_to_next_start_switch_pct"] == pytest.approx(0.5)
