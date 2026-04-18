import math
from collections import Counter


FEATURE_NAMES = [
    "entropy",
    "markov_entropy",
    "kl_divergence",
    "longest_run",
    "alternation_rate",
    "balance_deviation",
    "lag1_autocorrelation",
    "run_count",
    "mean_run_length",
    "alternation_deviation",
    "longest_alternating_run",
    "near_alternation_score",
    "pattern_break_rate",
]


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


def ones_fraction(sequence):
    return sequence.count('1') / len(sequence)


def balance_deviation(sequence):
    return abs(ones_fraction(sequence) - 0.5)


def lag1_autocorrelation(sequence):
    values = [-1 if bit == '0' else 1 for bit in sequence]
    x = values[:-1]
    y = values[1:]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denom_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    denom_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


def run_count(sequence):
    return 1 + sum(
        1 for i in range(1, len(sequence))
        if sequence[i] != sequence[i-1]
    )


def mean_run_length(sequence):
    return len(sequence) / run_count(sequence)


def alternation_deviation(sequence):
    return abs(alternation_rate(sequence) - 0.5)


def longest_alternating_run(sequence):
    max_run = 1
    current = 1

    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1

    return max_run


def near_alternation_score(sequence):
    pattern_zero = ''.join('0' if i % 2 == 0 else '1' for i in range(len(sequence)))
    pattern_one = ''.join('1' if i % 2 == 0 else '0' for i in range(len(sequence)))
    matches_zero = sum(a == b for a, b in zip(sequence, pattern_zero))
    matches_one = sum(a == b for a, b in zip(sequence, pattern_one))

    return max(matches_zero, matches_one) / len(sequence)


def pattern_break_rate(sequence):
    repeats = sum(
        1 for i in range(1, len(sequence))
        if sequence[i] == sequence[i-1]
    )
    return repeats / (len(sequence) - 1)


def extract_features(sequence):
    return [
        entropy(sequence),
        markov_entropy(sequence),
        kl_divergence(sequence),
        longest_run(sequence),
        alternation_rate(sequence),
        balance_deviation(sequence),
        lag1_autocorrelation(sequence),
        run_count(sequence),
        mean_run_length(sequence),
        alternation_deviation(sequence),
        longest_alternating_run(sequence),
        near_alternation_score(sequence),
        pattern_break_rate(sequence),
    ]


def extract_feature_dict(sequence):
    return dict(zip(FEATURE_NAMES, extract_features(sequence)))
