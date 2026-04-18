import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_PATH = PROJECT_ROOT / "analytics.csv"
SUPABASE_TABLE = "analytics"

REQUIRED_COLUMNS = {"sequence", "actual_label"}
LABEL_MAP = {"Random": 0, "Human": 1}
TARGET_NAMES = ["random", "human"]
SKIPPED_TEMPLATE = {
    "short_sequence": 0,
    "non_binary_sequence": 0,
    "invalid_label": 0,
    "duplicate_sequence": 0,
}


def validate_required_columns(df):
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing:
        raise ValueError(f"Missing required analytics columns: {', '.join(missing)}")


def clean_sequence(sequence):
    return "".join(str(sequence).split())


def prepare_labeled_dataframe(df):
    validate_required_columns(df)

    rows = []
    seen_sequences = set()
    skipped = SKIPPED_TEMPLATE.copy()

    for _, row in df.iterrows():
        label = row["actual_label"]
        sequence = row["sequence"]

        if not isinstance(label, str) or label.strip() not in LABEL_MAP:
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

        if sequence in seen_sequences:
            skipped["duplicate_sequence"] += 1
            continue

        seen_sequences.add(sequence)
        label = label.strip()
        rows.append(
            {
                "sequence": sequence,
                "actual_label": label,
                "label": LABEL_MAP[label],
                "session_id": row.get("session_id"),
                "batch_id": row.get("batch_id"),
                "batch_position": row.get("batch_position"),
            }
        )

    return pd.DataFrame(rows), skipped


def prepare_labeled_rows(df):
    labeled_df, skipped = prepare_labeled_dataframe(df)

    if labeled_df.empty:
        return [], np.array([], dtype=int), skipped

    return (
        labeled_df["sequence"].tolist(),
        labeled_df["label"].to_numpy(dtype=int),
        skipped,
    )


def select_group_ids(df):
    for column, mode in [("batch_id", "batch_id"), ("session_id", "session_id")]:
        if column in df.columns and df[column].notna().any():
            group_ids = df[column].astype("string")
            fallback = pd.Series(
                [f"row_{index}" for index in df.index],
                index=df.index,
                dtype="string",
            )
            return mode, group_ids.where(group_ids.notna(), fallback)

    return (
        "row_level_fallback",
        pd.Series([f"row_{index}" for index in df.index], index=df.index, dtype="string"),
    )


def summarize_groups(df):
    return {
        "rows_with_session_id": int(df["session_id"].notna().sum()) if "session_id" in df else 0,
        "session_count": int(df["session_id"].dropna().nunique()) if "session_id" in df else 0,
        "rows_with_batch_id": int(df["batch_id"].notna().sum()) if "batch_id" in df else 0,
        "batch_count": int(df["batch_id"].dropna().nunique()) if "batch_id" in df else 0,
    }


def class_counts(labels):
    labels = pd.Series(labels)
    return {
        "Random": int((labels == 0).sum()),
        "Human": int((labels == 1).sum()),
    }


def get_supabase_config():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

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


def load_real_dataframe():
    supabase_config = get_supabase_config()

    if supabase_config:
        return load_supabase_dataframe(supabase_config)

    if ANALYTICS_PATH.exists():
        return load_csv_dataframe()

    raise ValueError(
        "No analytics.csv found and SUPABASE_SERVICE_ROLE_KEY is not set."
    )


def missing_private_supabase_key():
    return bool(os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_KEY"))
