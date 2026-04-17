import pytest

from features import (
    alternation_rate,
    entropy,
    extract_features,
    kl_divergence,
    longest_run,
    markov_entropy,
)


def test_extract_features_returns_five_numeric_values():
    features = extract_features("0011010101")

    assert len(features) == 5
    assert all(isinstance(value, (int, float)) for value in features)


def test_all_zero_sequence_features():
    sequence = "0000000000"

    assert entropy(sequence) == pytest.approx(0.0)
    assert markov_entropy(sequence) == pytest.approx(0.0)
    assert kl_divergence(sequence) == pytest.approx(1.0)
    assert longest_run(sequence) == len(sequence)
    assert alternation_rate(sequence) == pytest.approx(0.0)


def test_perfect_alternation_features():
    sequence = "0101010101"

    assert entropy(sequence) == pytest.approx(1.0)
    assert longest_run(sequence) == 1
    assert alternation_rate(sequence) == pytest.approx(1.0)


def test_balanced_sequence_has_zero_kl_divergence():
    assert kl_divergence("00110011") == pytest.approx(0.0)
