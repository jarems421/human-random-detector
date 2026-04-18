from collections import Counter

from features import extract_features, ones_fraction


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


def explain_sequence(sequence, max_signals=5):
    features = extract_feature_dict(sequence)
    motif, motif_score = strongest_motif(sequence)

    if looks_random_like(features):
        return [
            {
                "tag": "random_like",
                "title": "Random-like",
                "message": "This sequence has a mix of balance, streaks, and switches that looks harder to separate.",
                "strength": 0.4,
            }
        ]

    signals = []

    if features["alternation_rate"] >= 0.7 or features["near_alternation_score"] >= 0.8:
        signals.append(
            {
                "tag": "alternation_bias",
                "title": "Alternation bias",
                "message": "You switched between 0 and 1 more often than true random usually does.",
                "strength": max(features["alternation_rate"], features["near_alternation_score"]),
            }
        )

    if len(sequence) >= 20 and features["longest_run"] <= 2:
        signals.append(
            {
                "tag": "streak_avoidance",
                "title": "Streak avoidance",
                "message": "The longest streak is very short, which often happens when people avoid runs.",
                "strength": 1 - (features["longest_run"] / 6),
            }
        )
    elif len(sequence) >= 20 and features["longest_run"] <= 3 and features["alternation_rate"] > 0.55:
        signals.append(
            {
                "tag": "streak_avoidance",
                "title": "Streak avoidance",
                "message": "The sequence has short runs and frequent switching.",
                "strength": 0.55,
            }
        )

    if features["balance_deviation"] <= 0.08 and features["longest_run"] <= 4:
        signals.append(
            {
                "tag": "balance_seeking",
                "title": "Balance seeking",
                "message": "The number of 0s and 1s is almost perfectly balanced.",
                "strength": 1 - features["balance_deviation"],
            }
        )

    if motif is not None and motif_score >= 0.7:
        signals.append(
            {
                "tag": "repeated_motif",
                "title": "Repeated motif",
                "message": f"The sequence leans on the repeated motif {motif}.",
                "strength": motif_score,
            }
        )

    ones = ones_fraction(sequence)
    if ones >= 0.6 or ones <= 0.4:
        bit = "1" if ones >= 0.6 else "0"
        signals.append(
            {
                "tag": "bit_bias",
                "title": "Soft bit bias",
                "message": f"The sequence favors {bit}s more than a balanced random sample would.",
                "strength": abs(ones - 0.5) * 2,
            }
        )

    if not signals:
        signals.append(
            {
                "tag": "random_like",
                "title": "Random-like",
                "message": "This sequence has a mix of balance, streaks, and switches that looks harder to separate.",
                "strength": 0.4,
            }
        )

    deduped = dedupe_signals(signals)
    deduped.sort(key=lambda signal: signal["strength"], reverse=True)
    return deduped[:max_signals]


def explanation_tags(sequence, max_signals=5):
    return [signal["tag"] for signal in explain_sequence(sequence, max_signals=max_signals)]


def feature_rows(sequence):
    features = extract_feature_dict(sequence)
    return [
        {"Feature": name.replace("_", " ").title(), "Value": value}
        for name, value in features.items()
    ]


def strongest_motif(sequence):
    if len(sequence) < 8:
        return None, 0.0

    best_motif = None
    best_score = 0.0

    for width in range(2, 5):
        chunks = [
            sequence[index:index + width]
            for index in range(0, len(sequence) - width + 1, width)
        ]

        if len(chunks) < 3:
            continue

        motif, count = Counter(chunks).most_common(1)[0]
        score = count / len(chunks)

        if score > best_score:
            best_motif = motif
            best_score = score

    return best_motif, best_score


def looks_random_like(features):
    return (
        features["entropy"] >= 0.95
        and 0.4 <= features["alternation_rate"] <= 0.65
        and features["balance_deviation"] <= 0.15
        and 3 <= features["longest_run"] <= 7
    )


def dedupe_signals(signals):
    by_tag = {}

    for signal in signals:
        existing = by_tag.get(signal["tag"])

        if existing is None or signal["strength"] > existing["strength"]:
            by_tag[signal["tag"]] = signal

    return list(by_tag.values())


def extract_feature_dict(sequence):
    return dict(zip(FEATURE_NAMES, extract_features(sequence)))
