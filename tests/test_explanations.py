from explanations import explain_sequence, explanation_tags


def tags_for(sequence):
    return explanation_tags(sequence, max_signals=5)


def test_near_alternating_sequence_gets_alternation_and_streak_signals():
    tags = tags_for("01010101010101010101")

    assert "alternation_bias" in tags
    assert "streak_avoidance" in tags


def test_balanced_short_run_sequence_gets_balance_and_streak_signals():
    tags = tags_for("00110011001100110011")

    assert "balance_seeking" in tags
    assert "streak_avoidance" in tags


def test_biased_sequence_gets_bit_bias_signal():
    tags = tags_for("11101110111011101110")

    assert "bit_bias" in tags


def test_random_looking_sequence_does_not_overstate_human_biases():
    tags = tags_for("00101110100111010010")

    assert tags == ["random_like"]


def test_explanations_have_user_facing_text():
    signals = explain_sequence("01010101010101010101")

    assert all(signal["title"] for signal in signals)
    assert all(signal["message"] for signal in signals)
