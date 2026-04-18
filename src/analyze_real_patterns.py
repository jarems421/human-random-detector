import json
import os
from pathlib import Path

import pandas as pd
import requests

from features import alternation_rate, longest_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_PATH = PROJECT_ROOT / "analytics.csv"
REPORT_PATH = PROJECT_ROOT / "real_pattern_analysis.json"
SUPABASE_TABLE = "analytics"

REQUIRED_COLUMNS = {"sequence", "actual_label"}
VALID_LABELS = {"Human", "Random"}


def clean_sequence(sequence):
    return "".join(str(sequence).split())


def validate_required_columns(df):
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing:
        raise ValueError(f"Missing required analytics columns: {', '.join(missing)}")


def prepare_valid_rows(df):
    validate_required_columns(df)

    rows = []
    skipped = {
        "short_sequence": 0,
        "non_binary_sequence": 0,
        "invalid_label": 0,
    }

    for _, row in df.iterrows():
        label = row["actual_label"]
        sequence = row["sequence"]

        if not isinstance(label, str) or label.strip() not in VALID_LABELS:
            skipped["invalid_label"] += 1
            continue

        if not isinstance(sequence, str):
            skipped["non_binary_sequence"] += 1
            continue

        sequence = clean_sequence(sequence)

        if len(sequence) < 10:
            skipped["short_sequence"] += 1
            continue

        if not all(char in "01" for char in sequence):
            skipped["non_binary_sequence"] += 1
            continue

        rows.append(
            {
                "sequence": sequence,
                "actual_label": label.strip(),
                "session_id": row.get("session_id"),
                "batch_id": row.get("batch_id"),
                "batch_position": row.get("batch_position"),
            }
        )

    return pd.DataFrame(rows), skipped


def starts_with_zero(sequence):
    return int(sequence[0] == "0")


def longest_alternating_run(sequence):
    max_run = 1
    current_run = 1

    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    return max_run


def near_alternation_score(sequence):
    pattern_zero = "".join("0" if i % 2 == 0 else "1" for i in range(len(sequence)))
    pattern_one = "".join("1" if i % 2 == 0 else "0" for i in range(len(sequence)))

    matches_zero = sum(a == b for a, b in zip(sequence, pattern_zero))
    matches_one = sum(a == b for a, b in zip(sequence, pattern_one))

    return max(matches_zero, matches_one) / len(sequence)


def pattern_break_count(sequence):
    return sum(sequence[i] == sequence[i - 1] for i in range(1, len(sequence)))


def add_pattern_metrics(df):
    df = df.copy()
    df["starts_with_zero"] = df["sequence"].apply(starts_with_zero)
    df["alternation_rate"] = df["sequence"].apply(alternation_rate)
    df["longest_run"] = df["sequence"].apply(longest_run)
    df["longest_alternating_run"] = df["sequence"].apply(longest_alternating_run)
    df["near_alternation_score"] = df["sequence"].apply(near_alternation_score)
    df["pattern_break_count"] = df["sequence"].apply(pattern_break_count)
    return df


def summarize_by_label(df):
    metrics = [
        "starts_with_zero",
        "alternation_rate",
        "longest_run",
        "longest_alternating_run",
        "near_alternation_score",
        "pattern_break_count",
    ]

    summary = {}

    for label, group in df.groupby("actual_label"):
        summary[label] = {"count": int(len(group))}

        for metric in metrics:
            summary[label][f"mean_{metric}"] = float(group[metric].mean())

    return summary


def summarize_batches(df):
    required = {"actual_label", "sequence", "batch_id", "batch_position"}

    if not required.issubset(df.columns):
        return {
            "tracked_batch_count": 0,
            "rows_with_batch_metadata": 0,
            "human_batch_count": 0,
            "mean_human_batch_size": 0.0,
            "human_batches_with_both_start_bits_pct": 0.0,
            "human_adjacent_start_bit_switch_pct": 0.0,
        }

    batch_df = df.dropna(subset=["batch_id", "batch_position"]).copy()
    batch_df = batch_df[batch_df["actual_label"] == "Human"]

    if batch_df.empty:
        return {
            "tracked_batch_count": 0,
            "rows_with_batch_metadata": 0,
            "human_batch_count": 0,
            "mean_human_batch_size": 0.0,
            "human_batches_with_both_start_bits_pct": 0.0,
            "human_adjacent_start_bit_switch_pct": 0.0,
        }

    batch_df["batch_position"] = pd.to_numeric(batch_df["batch_position"], errors="coerce")
    batch_df = batch_df.dropna(subset=["batch_position"])
    batch_df["start_bit"] = batch_df["sequence"].str[0]

    grouped = batch_df.sort_values(["batch_id", "batch_position"]).groupby("batch_id")
    batch_sizes = grouped.size()
    batches_with_both_bits = grouped["start_bit"].nunique().gt(1)

    adjacent_pairs = 0
    adjacent_switches = 0

    for _, group in grouped:
        start_bits = group["start_bit"].tolist()

        for previous, current in zip(start_bits, start_bits[1:]):
            adjacent_pairs += 1

            if previous != current:
                adjacent_switches += 1

    return {
        "tracked_batch_count": int(df["batch_id"].dropna().nunique()),
        "rows_with_batch_metadata": int(df["batch_id"].notna().sum()),
        "human_batch_count": int(len(batch_sizes)),
        "mean_human_batch_size": float(batch_sizes.mean()),
        "human_batches_with_both_start_bits_pct": float(batches_with_both_bits.mean()),
        "human_adjacent_start_bit_switch_pct": (
            float(adjacent_switches / adjacent_pairs) if adjacent_pairs else 0.0
        ),
    }


def analyze_dataframe(df):
    valid_df, skipped = prepare_valid_rows(df)

    if valid_df.empty:
        raise ValueError("No valid labeled rows found. Collect labeled samples in the app first.")

    metric_df = add_pattern_metrics(valid_df)

    return {
        "valid_rows": int(len(metric_df)),
        "skipped_rows": int(sum(skipped.values())),
        "skipped": skipped,
        "label_summary": summarize_by_label(metric_df),
        "batch_summary": summarize_batches(metric_df),
    }


def print_analysis(analysis):
    print("Real pattern analysis complete")
    print(f"Valid rows: {analysis['valid_rows']}")
    print(f"Skipped rows: {analysis['skipped_rows']}")
    print(f"Skipped detail: {analysis['skipped']}")
    print("\nLabel summary:")
    print(pd.DataFrame(analysis["label_summary"]).transpose())
    print("\nBatch summary:")

    for key, value in analysis["batch_summary"].items():
        print(f"{key}: {value}")


def get_supabase_config():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        return None

    return {
        "url": url.rstrip("/"),
        "key": key,
    }


def get_supabase_headers(config):
    return {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
    }


def load_supabase_dataframe(config):
    endpoint = f"{config['url']}/rest/v1/{SUPABASE_TABLE}"
    params = {
        "select": "sequence,actual_label,session_id,batch_id,batch_position",
        "order": "created_at.asc",
    }
    response = requests.get(
        endpoint,
        headers=get_supabase_headers(config),
        params=params,
        timeout=10,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


def load_csv_dataframe():
    return pd.read_csv(
        ANALYTICS_PATH,
        dtype={
            "sequence": "string",
            "actual_label": "string",
            "session_id": "string",
            "batch_id": "string",
        },
    )


def main():
    try:
        supabase_config = get_supabase_config()

        if supabase_config:
            df = load_supabase_dataframe(supabase_config)
        elif ANALYTICS_PATH.exists():
            df = load_csv_dataframe()
        else:
            print("No analytics.csv found. Collect labeled samples in the app first.")
            return 0

        analysis = analyze_dataframe(df)
    except ValueError as exc:
        print(exc)
        return 1
    except requests.RequestException as exc:
        print(f"Could not load Supabase analytics: {exc}")
        return 1

    REPORT_PATH.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print_analysis(analysis)
    print(f"Saved real pattern analysis to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
