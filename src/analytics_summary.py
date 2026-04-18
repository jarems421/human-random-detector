import pandas as pd


PUBLIC_SUMMARY_COLUMNS = [
    "total_rows",
    "human_rows",
    "random_rows",
    "model_accuracy",
    "human_precision",
    "human_recall",
    "random_precision",
    "random_recall",
    "avg_p_human_for_human",
    "avg_p_human_for_random",
    "guessed_rows",
    "user_accuracy",
]


def empty_public_summary():
    return {
        "total_rows": 0,
        "human_rows": 0,
        "random_rows": 0,
        "model_accuracy": None,
        "human_precision": None,
        "human_recall": None,
        "random_precision": None,
        "random_recall": None,
        "avg_p_human_for_human": None,
        "avg_p_human_for_random": None,
        "guessed_rows": 0,
        "user_accuracy": None,
    }


def build_public_summary(df):
    if df is None or df.empty:
        return empty_public_summary()

    df = df.copy()
    df["p_human"] = pd.to_numeric(df.get("p_human"), errors="coerce")
    labels = df["actual_label"]
    predictions = df["model_prediction"]
    human_rows = labels == "Human"
    random_rows = labels == "Random"
    guessed_rows = df.get("user_guess", pd.Series(index=df.index, dtype="string")).isin(
        ["Human", "Random"]
    )

    return {
        "total_rows": int(len(df)),
        "human_rows": int(human_rows.sum()),
        "random_rows": int(random_rows.sum()),
        "model_accuracy": safe_mean(predictions == labels),
        "human_precision": precision(labels, predictions, "Human"),
        "human_recall": recall(labels, predictions, "Human"),
        "random_precision": precision(labels, predictions, "Random"),
        "random_recall": recall(labels, predictions, "Random"),
        "avg_p_human_for_human": safe_mean(df.loc[human_rows, "p_human"]),
        "avg_p_human_for_random": safe_mean(df.loc[random_rows, "p_human"]),
        "guessed_rows": int(guessed_rows.sum()),
        "user_accuracy": safe_mean(df.loc[guessed_rows, "user_guess"] == labels.loc[guessed_rows]),
    }


def summary_from_supabase_row(row):
    summary = empty_public_summary()

    for column in PUBLIC_SUMMARY_COLUMNS:
        if column in row and pd.notna(row[column]):
            summary[column] = row[column]

    for column in ["total_rows", "human_rows", "random_rows", "guessed_rows"]:
        summary[column] = int(summary[column] or 0)

    for column in set(PUBLIC_SUMMARY_COLUMNS) - {
        "total_rows",
        "human_rows",
        "random_rows",
        "guessed_rows",
    }:
        summary[column] = None if summary[column] is None else float(summary[column])

    return summary


def label_count_frame(summary):
    return pd.DataFrame(
        [
            {"label": "Human", "rows": summary["human_rows"]},
            {"label": "Random", "rows": summary["random_rows"]},
        ]
    )


def probability_by_label_frame(summary):
    rows = []

    if summary["avg_p_human_for_human"] is not None:
        rows.append({"label": "Human", "avg_p_human": summary["avg_p_human_for_human"]})

    if summary["avg_p_human_for_random"] is not None:
        rows.append({"label": "Random", "avg_p_human": summary["avg_p_human_for_random"]})

    return pd.DataFrame(rows)


def safe_mean(values):
    if values is None or len(values) == 0:
        return None

    value = values.mean()
    return None if pd.isna(value) else float(value)


def precision(labels, predictions, positive_label):
    predicted_positive = predictions == positive_label

    if not predicted_positive.any():
        return None

    true_positive = (labels[predicted_positive] == positive_label).sum()
    return float(true_positive / predicted_positive.sum())


def recall(labels, predictions, positive_label):
    actual_positive = labels == positive_label

    if not actual_positive.any():
        return None

    true_positive = (predictions[actual_positive] == positive_label).sum()
    return float(true_positive / actual_positive.sum())
