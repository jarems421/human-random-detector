import pytest

from features import (
    FEATURE_NAMES,
    alternation_rate,
    balance_deviation,
    entropy,
    extract_feature_dict,
    extract_features,
    kl_divergence,
    lag1_autocorrelation,
    longest_alternating_run,
    longest_run,
    markov_entropy,
    mean_run_length,
    near_alternation_score,
    pattern_break_rate,
    run_count,
)


def test_feature_schema_contract_stays_in_sync():
    sequence = "0011010101"
    features = extract_features(sequence)
    feature_dict = extract_feature_dict(sequence)

    assert len(FEATURE_NAMES) == 13
    assert len(features) == len(FEATURE_NAMES)
    assert list(feature_dict) == FEATURE_NAMES


def test_extract_features_returns_thirteen_numeric_values():
    features = extract_features("0011010101")

    assert len(features) == 13
    assert all(isinstance(value, (int, float)) for value in features)


def test_all_zero_sequence_features():
    sequence = "0000000000"

    assert entropy(sequence) == pytest.approx(0.0)
    assert markov_entropy(sequence) == pytest.approx(0.0)
    assert kl_divergence(sequence) == pytest.approx(1.0)
    assert longest_run(sequence) == len(sequence)
    assert alternation_rate(sequence) == pytest.approx(0.0)
    assert balance_deviation(sequence) == pytest.approx(0.5)
    assert run_count(sequence) == 1
    assert mean_run_length(sequence) == len(sequence)


def test_perfect_alternation_features():
    sequence = "0101010101"

    assert entropy(sequence) == pytest.approx(1.0)
    assert longest_run(sequence) == 1
    assert alternation_rate(sequence) == pytest.approx(1.0)
    assert lag1_autocorrelation(sequence) < 0
    assert longest_alternating_run(sequence) == len(sequence)
    assert near_alternation_score(sequence) == pytest.approx(1.0)


def test_balanced_sequence_has_zero_kl_divergence():
    assert kl_divergence("00110011") == pytest.approx(0.0)


def test_balanced_sequence_has_zero_balance_deviation():
    assert balance_deviation("00110011") == pytest.approx(0.0)


def test_all_one_sequence_has_max_balance_deviation():
    assert balance_deviation("1111111111") == pytest.approx(0.5)


def test_repeated_sequence_run_structure():
    sequence = "1111111111"

    assert run_count(sequence) == 1
    assert mean_run_length(sequence) == pytest.approx(len(sequence))


def test_near_alternation_score_for_repeated_values():
    assert near_alternation_score("0000000000") == pytest.approx(0.5)


def test_pattern_break_rate_counts_repeated_adjacent_pairs():
    sequence = "0101101010"
    repeated_pairs = 1

    assert pattern_break_rate(sequence) == pytest.approx(repeated_pairs / (len(sequence) - 1))
