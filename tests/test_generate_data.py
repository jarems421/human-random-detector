import random

import generate_data
from generate_data import (
    create_dataset,
    generate_human_like,
    generate_true_random,
    human_balanced_streak_avoidant,
    human_chunk_pattern,
    human_near_alternating,
    human_noisy,
    human_soft_biased,
)


def assert_binary_sequence(sequence, length):
    assert len(sequence) == length
    assert set(sequence) <= {'0', '1'}


def test_create_dataset_returns_two_classes_per_sample():
    data, labels = create_dataset(n=12, length=50)

    assert len(data) == 24
    assert len(labels) == 24


def test_create_dataset_sequences_have_requested_length_and_binary_values():
    data, _ = create_dataset(n=12, length=37)

    for sequence in data:
        assert_binary_sequence(sequence, 37)


def test_create_dataset_labels_remain_random_then_human():
    _, labels = create_dataset(n=5, length=20)

    assert labels == [0, 1] * 5


def test_generate_human_like_produces_valid_binary_sequences():
    random.seed(42)

    for _ in range(50):
        assert_binary_sequence(generate_human_like(length=31), 31)


def test_generate_human_like_uses_weighted_behaviour_sampling(monkeypatch):
    captured = {}

    def fake_choices(behaviours, weights, k):
        captured["behaviours"] = behaviours
        captured["weights"] = weights
        captured["k"] = k
        return [generate_data.human_noisy]

    monkeypatch.setattr(generate_data.random, "choices", fake_choices)

    assert_binary_sequence(generate_human_like(length=17), 17)
    assert captured["behaviours"] == [
        generate_data.human_near_alternating,
        generate_data.human_balanced_streak_avoidant,
        generate_data.human_chunk_pattern,
        generate_data.human_soft_biased,
        generate_data.human_noisy,
    ]
    assert captured["weights"] == [35, 25, 20, 10, 10]
    assert captured["k"] == 1


def test_all_generators_produce_valid_binary_sequences():
    random.seed(42)
    generators = [
        generate_true_random,
        human_near_alternating,
        human_balanced_streak_avoidant,
        human_chunk_pattern,
        human_soft_biased,
        human_noisy,
    ]

    for generator in generators:
        assert_binary_sequence(generator(length=29), 29)
