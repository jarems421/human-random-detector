import math
from collections import Counter


def entropy(sequence):
    counts = Counter(sequence)
    probs = [count / len(sequence) for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def markov_entropy(sequence):
    transitions = {'00': 0, '01': 0, '10': 0, '11': 0}

    for i in range(len(sequence) - 1):
        transitions[sequence[i:i+2]] += 1

    total = sum(transitions.values())
    probs = [v / total for v in transitions.values() if v > 0]

    return -sum(p * math.log2(p) for p in probs)


def kl_divergence(sequence):
    p0 = sequence.count('0') / len(sequence)
    p1 = sequence.count('1') / len(sequence)

    eps = 1e-10
    p0 = max(p0, eps)
    p1 = max(p1, eps)

    return p0 * math.log2(p0 / 0.5) + p1 * math.log2(p1 / 0.5)


def longest_run(sequence):
    max_run = 1
    current = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1

    return max_run


def alternation_rate(sequence):
    changes = sum(
        1 for i in range(1, len(sequence))
        if sequence[i] != sequence[i-1]
    )
    return changes / (len(sequence) - 1)


def extract_features(sequence):
    return [
        entropy(sequence),
        markov_entropy(sequence),
        kl_divergence(sequence),
        longest_run(sequence),
        alternation_rate(sequence),
    ]