import pandas as pd
import pytest

from real_data import prepare_labeled_dataframe, select_group_ids


def test_prepare_labeled_dataframe_cleans_maps_and_deduplicates_rows():
    df = pd.DataFrame(
        [
            {"sequence": "01010 10101", "actual_label": "Human", "batch_id": "b1"},
            {"sequence": "0101010101", "actual_label": "Human", "batch_id": "b1"},
            {"sequence": "0011001100", "actual_label": "Random", "batch_id": "b2"},
        ]
    )

    labeled_df, skipped = prepare_labeled_dataframe(df)

    assert labeled_df["sequence"].tolist() == ["0101010101", "0011001100"]
    assert labeled_df["label"].tolist() == [1, 0]
    assert skipped["duplicate_sequence"] == 1


def test_prepare_labeled_dataframe_tracks_invalid_rows():
    df = pd.DataFrame(
        [
            {"sequence": "0101", "actual_label": "Human"},
            {"sequence": "0101010102", "actual_label": "Random"},
            {"sequence": "0101010101", "actual_label": "Unknown"},
        ]
    )

    labeled_df, skipped = prepare_labeled_dataframe(df)

    assert labeled_df.empty
    assert skipped["short_sequence"] == 1
    assert skipped["non_binary_sequence"] == 1
    assert skipped["invalid_label"] == 1


def test_prepare_labeled_dataframe_requires_sequence_and_label_columns():
    with pytest.raises(ValueError, match="actual_label"):
        prepare_labeled_dataframe(pd.DataFrame([{"sequence": "0101010101"}]))


def test_select_group_ids_prefers_batch_id_then_session_id_then_rows():
    batch_df = pd.DataFrame(
        [
            {"batch_id": "b1", "session_id": "s1"},
            {"batch_id": "b2", "session_id": "s1"},
        ]
    )
    session_df = pd.DataFrame(
        [
            {"batch_id": None, "session_id": "s1"},
            {"batch_id": None, "session_id": "s2"},
        ]
    )
    row_df = pd.DataFrame([{"batch_id": None, "session_id": None}])

    assert select_group_ids(batch_df)[0] == "batch_id"
    assert select_group_ids(session_df)[0] == "session_id"
    assert select_group_ids(row_df)[0] == "row_level_fallback"
