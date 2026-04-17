import pytest

from predict import predict


def test_predict_rejects_short_sequences():
    with pytest.raises(ValueError, match="at least 10"):
        predict("010101")


def test_predict_rejects_non_binary_sequences():
    with pytest.raises(ValueError, match="only 0s and 1s"):
        predict("0101010102")


def test_predict_returns_random_and_human_probabilities():
    probabilities = predict("01010101010101010101")

    assert len(probabilities) == 2
    assert probabilities[0] >= 0
    assert probabilities[1] >= 0
    assert sum(probabilities) == pytest.approx(1.0)
