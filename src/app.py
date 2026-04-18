from datetime import datetime
from pathlib import Path
import json
import os
import secrets
import uuid

import joblib
import pandas as pd
import requests
import streamlit as st

from analytics_summary import (
    build_public_summary,
    empty_public_summary,
    label_count_frame,
    probability_by_label_frame,
    summary_from_supabase_row,
)
from explanations import explain_sequence, explanation_tags, feature_rows
from features import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
ANALYTICS_PATH = PROJECT_ROOT / "analytics.csv"
BATCH_SIZE = 5
CHALLENGE_ROUNDS = 5
SEQUENCE_LENGTH = 50
MODEL_VERSION = "synthetic-human-v2"
SUPABASE_TABLE = "analytics"
PUBLIC_SUMMARY_VIEW = "analytics_public_summary"
REQUIRED_ANALYTICS_COLUMNS = [
    "actual_label",
    "model_prediction",
    "user_guess",
    "p_human",
]


st.set_page_config(
    page_title="Human Randomness Experiment",
    page_icon="01",
    layout="wide",
)


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def generate_random_sequence(length=SEQUENCE_LENGTH):
    return "".join(secrets.choice(["0", "1"]) for _ in range(length))


def get_secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def get_supabase_config():
    url = get_secret("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

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
        "Content-Type": "application/json",
    }


def supabase_enabled():
    return get_supabase_config() is not None


def clean_sequence(sequence):
    return "".join(str(sequence).split())


def validate_sequence(sequence):
    sequence = clean_sequence(sequence)

    if len(sequence) < 10:
        return sequence, "Enter at least 10 bits."

    if not all(char in "01" for char in sequence):
        return sequence, "Use only 0s and 1s."

    return sequence, None


def predict_sequence(sequence):
    features = extract_features(sequence)
    scaled_features = scaler.transform([features])
    p_random, p_human = model.predict_proba(scaled_features)[0]
    prediction = "Human" if p_human > p_random else "Random"

    return {
        "prediction": prediction,
        "p_random": float(p_random),
        "p_human": float(p_human),
        "confidence": float(max(p_random, p_human)),
        "explanations": explain_sequence(sequence),
        "explanation_tags": explanation_tags(sequence),
    }


def log_result(
    sequence,
    actual_label,
    p_human,
    p_random,
    model_prediction,
    user_guess,
    session_id,
    batch_id,
    batch_position,
    source_mode,
    tags,
):
    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sequence": sequence,
        "actual_label": actual_label,
        "p_human": p_human,
        "p_random": p_random,
        "model_prediction": model_prediction,
        "user_guess": user_guess,
        "session_id": session_id,
        "batch_id": batch_id,
        "batch_position": batch_position,
        "model_version": MODEL_VERSION,
        "sequence_length": len(sequence),
        "source_mode": source_mode,
        "explanation_tags": tags,
    }

    if supabase_enabled():
        insert_supabase_result(data)
        return

    append_csv_result(data)


def insert_supabase_result(data):
    config = get_supabase_config()
    endpoint = f"{config['url']}/rest/v1/{SUPABASE_TABLE}"
    headers = get_supabase_headers(config)
    headers["Prefer"] = "return=minimal"
    response = requests.post(endpoint, headers=headers, json=supabase_payload(data), timeout=10)

    if response.status_code < 400:
        return

    legacy_response = requests.post(
        endpoint,
        headers=headers,
        json=supabase_payload(data, include_metadata=False),
        timeout=10,
    )

    if legacy_response.status_code >= 400:
        raise requests.HTTPError(
            f"{legacy_response.status_code} {legacy_response.reason}: {legacy_response.text}",
            response=legacy_response,
        )


def supabase_payload(data, include_metadata=True):
    payload = {
        "sequence": data["sequence"],
        "actual_label": data["actual_label"],
        "p_human": data["p_human"],
        "p_random": data["p_random"],
        "model_prediction": data["model_prediction"],
        "user_guess": data["user_guess"] or None,
        "session_id": data["session_id"],
        "batch_id": data["batch_id"],
        "batch_position": data["batch_position"],
    }

    if include_metadata:
        payload.update(
            {
                "model_version": data["model_version"],
                "sequence_length": data["sequence_length"],
                "source_mode": data["source_mode"],
                "explanation_tags": data["explanation_tags"],
            }
        )

    return payload


def append_csv_result(data):
    csv_data = data.copy()
    csv_data["explanation_tags"] = json.dumps(csv_data["explanation_tags"])
    df = pd.DataFrame([csv_data])

    if ANALYTICS_PATH.exists():
        df.to_csv(ANALYTICS_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(ANALYTICS_PATH, index=False)


def save_collected_sequence(
    sequence,
    actual_label,
    user_guess,
    batch_id,
    batch_position,
    source_mode="collect",
):
    result = predict_sequence(sequence)

    log_result(
        sequence=sequence,
        actual_label=actual_label,
        p_human=result["p_human"],
        p_random=result["p_random"],
        model_prediction=result["prediction"],
        user_guess=user_guess,
        session_id=st.session_state.session_id,
        batch_id=batch_id,
        batch_position=batch_position,
        source_mode=source_mode,
        tags=result["explanation_tags"],
    )

    return result


def load_public_analytics_summary():
    if supabase_enabled():
        return load_supabase_public_summary(), None

    df, missing_columns = load_csv_analytics()

    if df is None or missing_columns:
        return empty_public_summary(), missing_columns

    return build_public_summary(df), None


def load_supabase_public_summary():
    config = get_supabase_config()
    endpoint = f"{config['url']}/rest/v1/{PUBLIC_SUMMARY_VIEW}"
    response = requests.get(
        endpoint,
        headers=get_supabase_headers(config),
        params={"select": "*"},
        timeout=10,
    )
    response.raise_for_status()
    rows = response.json()

    if not rows:
        return empty_public_summary()

    return summary_from_supabase_row(rows[0])


def load_csv_analytics():
    if not ANALYTICS_PATH.exists():
        return None, None

    df = pd.read_csv(
        ANALYTICS_PATH,
        dtype={
            "timestamp": "string",
            "sequence": "string",
            "actual_label": "string",
            "model_prediction": "string",
            "user_guess": "string",
            "session_id": "string",
            "batch_id": "string",
            "source_mode": "string",
            "model_version": "string",
        },
    )

    return prepare_analytics_dataframe(df)


def prepare_analytics_dataframe(df):
    missing = [col for col in REQUIRED_ANALYTICS_COLUMNS if col not in df.columns]

    if missing:
        return df, missing

    df["p_human"] = pd.to_numeric(df["p_human"], errors="coerce")
    df["p_random"] = pd.to_numeric(df.get("p_random"), errors="coerce")

    return df, None


def show_probability_summary(result):
    st.metric("Model prediction", result["prediction"])
    st.metric("Confidence", f"{result['confidence']:.2f}")
    st.progress(result["p_human"], text=f"Human probability: {result['p_human']:.2f}")
    st.progress(result["p_random"], text=f"Random probability: {result['p_random']:.2f}")


def show_explanations(result):
    st.write("What gave it away")

    for signal in result["explanations"]:
        with st.container(border=True):
            st.write(f"**{signal['title']}**")
            st.caption(signal["message"])


def show_sequence_features(sequence):
    st.dataframe(pd.DataFrame(feature_rows(sequence)), hide_index=True, width="stretch")


def show_prediction_result(sequence, result):
    summary_col, explanation_col = st.columns([1, 2])

    with summary_col:
        show_probability_summary(result)

    with explanation_col:
        show_explanations(result)

    with st.expander("Numeric feature details"):
        show_sequence_features(sequence)


def ensure_session_defaults():
    st.session_state.setdefault("score", 0)
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("challenge_batch_id", str(uuid.uuid4()))
    st.session_state.setdefault("challenge_results", {})

    for i in range(BATCH_SIZE):
        st.session_state.setdefault(f"seq_{i}", "")
        st.session_state.setdefault(f"actual_{i}", "Human")
        st.session_state.setdefault(f"guess_{i}", "No guess")

    for i in range(CHALLENGE_ROUNDS):
        st.session_state.setdefault(f"challenge_seq_{i}", "")


def reset_challenge():
    st.session_state.challenge_batch_id = str(uuid.uuid4())
    st.session_state.challenge_results = {}

    for i in range(CHALLENGE_ROUNDS):
        st.session_state[f"challenge_seq_{i}"] = ""


def challenge_score():
    return sum(
        1
        for result in st.session_state.challenge_results.values()
        if result["prediction"] == "Random"
    )


def challenge_summary_text():
    results = list(st.session_state.challenge_results.values())

    if not results:
        return "Start with round 1 and try to make a human-made sequence look random."

    all_tags = [
        signal["tag"]
        for result in results
        for signal in result["explanations"]
        if signal["tag"] != "random_like"
    ]

    if all_tags:
        top_tag = pd.Series(all_tags).value_counts().index[0].replace("_", " ")
        return f"Your main giveaway was {top_tag}."

    return "Your attempts were fairly random-looking, so longer samples may be needed."


def show_challenge_tab():
    st.subheader("Beat the model")
    st.write("Enter five human-made bit sequences. You score when the model calls your sequence Random.")

    action_col_1, action_col_2 = st.columns([1, 1])

    with action_col_1:
        st.metric("Challenge score", f"{challenge_score()} / {CHALLENGE_ROUNDS}")

    with action_col_2:
        if st.button("Start a new challenge", width="stretch"):
            reset_challenge()
            st.rerun()

    for i in range(CHALLENGE_ROUNDS):
        with st.container(border=True):
            st.write(f"Round {i + 1}")
            sequence_input = st.text_input(
                "Your human-made sequence",
                key=f"challenge_seq_{i}",
                placeholder="Example: 01001101011000100110",
                help="Enter at least 10 bits. Spaces are ignored.",
                label_visibility="collapsed",
            )
            cleaned_sequence, error = validate_sequence(sequence_input)
            st.caption(f"Length after cleaning: {len(cleaned_sequence)} bits")

            if st.button("Score this round", key=f"score_challenge_{i}", width="stretch"):
                if error:
                    st.error(error)
                else:
                    try:
                        result = save_collected_sequence(
                            sequence=cleaned_sequence,
                            actual_label="Human",
                            user_guess=None,
                            batch_id=st.session_state.challenge_batch_id,
                            batch_position=i + 1,
                            source_mode="challenge",
                        )
                    except requests.RequestException as exc:
                        st.error(f"Could not save this round: {exc}")
                    else:
                        result["sequence"] = cleaned_sequence
                        st.session_state.challenge_results[i] = result
                        st.session_state.score = challenge_score()

            if i in st.session_state.challenge_results:
                result = st.session_state.challenge_results[i]
                fooled = result["prediction"] == "Random"

                if fooled:
                    st.success("Point scored. The model called this Random.")
                else:
                    st.info("The model spotted this as Human.")

                show_prediction_result(result["sequence"], result)

    if len(st.session_state.challenge_results) == CHALLENGE_ROUNDS:
        st.success(f"Final score: {challenge_score()} / {CHALLENGE_ROUNDS}")
        st.write(challenge_summary_text())


def show_analyze_tab():
    st.subheader("Analyze one sequence")
    st.write("Paste or type a sequence to see the prediction without saving it.")

    col_generate, col_clear = st.columns([1, 1])

    with col_generate:
        if st.button("Use known-random example", width="stretch"):
            st.session_state.try_sequence = generate_random_sequence()

    with col_clear:
        if st.button("Clear", key="clear_try_sequence", width="stretch"):
            st.session_state.try_sequence = ""

    sequence_input = st.text_area(
        "Sequence to analyze",
        key="try_sequence",
        height=110,
        placeholder="Example: 01001101011000100110",
        help="Enter at least 10 bits. Spaces and line breaks are okay.",
    )

    cleaned_sequence, error = validate_sequence(sequence_input)
    st.caption(f"Length after cleaning: {len(cleaned_sequence)} bits")

    if st.button("Analyze Sequence", type="primary", width="stretch"):
        if error:
            st.error(error)
        else:
            show_prediction_result(cleaned_sequence, predict_sequence(cleaned_sequence))


def show_collect_tab():
    st.subheader("Collect labeled examples")
    st.write("Use this when the source is known. The source label is saved for evaluation.")

    collection_mode = st.radio(
        "Known source",
        ["Human sequences", "Random sequences"],
        horizontal=True,
        help="Choose Human when the sequences were typed by a person. Choose Random for generated rows.",
    )

    if collection_mode == "Human sequences":
        show_human_collection()
    else:
        show_random_collection()

    show_advanced_collection()


def show_human_collection():
    human_input = st.text_area(
        "Paste one human-made sequence per line",
        height=220,
        placeholder="01001101011000100110\n10100100101101001010",
        help="Blank lines are ignored. Spaces inside a sequence are okay.",
    )
    st.caption("These rows help evaluate the model against real human behavior.")

    if st.button("Save Human Sequences", type="primary", width="stretch"):
        saved_rows = 0
        invalid_rows = 0
        batch_id = str(uuid.uuid4())
        batch_position = 0

        for line_number, raw_sequence in enumerate(human_input.splitlines(), start=1):
            if not raw_sequence.strip():
                continue

            sequence, error = validate_sequence(raw_sequence)

            if error:
                invalid_rows += 1
                st.warning(f"Line {line_number}: {error}")
                continue

            batch_position += 1

            try:
                result = save_collected_sequence(
                    sequence=sequence,
                    actual_label="Human",
                    user_guess=None,
                    batch_id=batch_id,
                    batch_position=batch_position,
                    source_mode="collection",
                )
            except requests.RequestException as exc:
                st.error(f"Could not save line {line_number}: {exc}")
                continue

            saved_rows += 1
            st.write(
                f"Line {line_number}: model said {result['prediction']} "
                f"with {result['confidence']:.2f} confidence."
            )

        show_save_outcome(saved_rows, invalid_rows, "human")


def show_random_collection():
    random_count = st.number_input(
        "Rows to generate",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Generated rows are automatically labeled Random.",
    )

    if st.button("Generate And Save Random Rows", type="primary", width="stretch"):
        saved_rows = 0
        batch_id = str(uuid.uuid4())

        for batch_position in range(1, random_count + 1):
            sequence = generate_random_sequence()

            try:
                save_collected_sequence(
                    sequence=sequence,
                    actual_label="Random",
                    user_guess=None,
                    batch_id=batch_id,
                    batch_position=batch_position,
                    source_mode="collection",
                )
            except requests.RequestException as exc:
                st.error(f"Could not save generated row: {exc}")
                continue

            saved_rows += 1

        show_save_outcome(saved_rows, 0, "random")


def show_save_outcome(saved_rows, invalid_rows, label):
    if saved_rows:
        destination = "Supabase" if supabase_enabled() else "analytics.csv"
        st.success(f"Saved {saved_rows} {label} row(s) to {destination}.")
    elif invalid_rows:
        st.error(f"No {label} rows were saved.")
    else:
        st.info("Enter at least one sequence first.")


def show_advanced_collection():
    with st.expander("Advanced: collect guesses for five examples"):
        st.write("Use this when you want to record a separate user guess before saving.")
        action_col_1, action_col_2 = st.columns([1, 1])

        with action_col_1:
            if st.button("Generate known-random batch", width="stretch"):
                for i in range(BATCH_SIZE):
                    st.session_state[f"seq_{i}"] = generate_random_sequence()
                    st.session_state[f"actual_{i}"] = "Random"
                    st.session_state[f"guess_{i}"] = "No guess"

        with action_col_2:
            if st.button("Clear batch", width="stretch"):
                for i in range(BATCH_SIZE):
                    st.session_state[f"seq_{i}"] = ""
                    st.session_state[f"actual_{i}"] = "Human"
                    st.session_state[f"guess_{i}"] = "No guess"

        st.divider()

        for i in range(BATCH_SIZE):
            st.text_input(
                f"Example {i + 1}",
                key=f"seq_{i}",
                placeholder="Type human data or use the random batch generator",
                help="At least 10 bits. Spaces are ignored.",
            )

            label_col, guess_col = st.columns([1, 1])

            with label_col:
                st.radio(
                    "Known source",
                    ["Human", "Random"],
                    key=f"actual_{i}",
                    horizontal=True,
                )

            with guess_col:
                st.radio(
                    "Your guess",
                    ["No guess", "Human", "Random"],
                    key=f"guess_{i}",
                    horizontal=True,
                )

        if st.button("Predict And Save Valid Rows", type="primary", width="stretch"):
            save_advanced_rows()


def save_advanced_rows():
    saved_rows = 0
    batch_id = str(uuid.uuid4())

    for i in range(BATCH_SIZE):
        sequence, error = validate_sequence(st.session_state[f"seq_{i}"])

        if error:
            st.warning(f"Example {i + 1}: {error}")
            continue

        actual_label = st.session_state[f"actual_{i}"]
        raw_user_guess = st.session_state[f"guess_{i}"]
        user_guess = None if raw_user_guess == "No guess" else raw_user_guess

        try:
            result = save_collected_sequence(
                sequence=sequence,
                actual_label=actual_label,
                user_guess=user_guess,
                batch_id=batch_id,
                batch_position=i + 1,
                source_mode="advanced_collection",
            )
        except requests.RequestException as exc:
            st.error(f"Could not save Example {i + 1}: {exc}")
            continue

        model_correct = result["prediction"] == actual_label
        user_correct = user_guess == actual_label if user_guess else None

        st.write(f"Example {i + 1}: `{sequence}`")
        st.write(f"Actual: {actual_label} | Model: {result['prediction']} | Your guess: {user_guess or 'No guess'}")
        st.write(f"Confidence: {result['confidence']:.2f}")

        if user_guess is None:
            st.info("Saved without a user guess.")
        elif user_correct and not model_correct:
            st.success("You beat the model on this one.")
            st.session_state.score += 1
        elif user_correct and model_correct:
            st.info("You and the model were both correct.")
        elif not user_correct and model_correct:
            st.error("The model was correct.")
        else:
            st.warning("Neither prediction matched the label.")

        saved_rows += 1

    if saved_rows:
        destination = "Supabase" if supabase_enabled() else "analytics.csv"
        st.success(f"Saved {saved_rows} labeled row(s) to {destination}.")
    else:
        st.error("No valid rows were saved.")


def show_analytics_tab():
    st.subheader("Aggregate analytics")
    st.write("Public analytics use aggregate counts and rates only. Raw submitted sequences are not shown here.")

    try:
        summary, missing_columns = load_public_analytics_summary()
    except requests.RequestException as exc:
        st.warning("Aggregate analytics are not ready yet.")

        if is_missing_public_summary_view(exc):
            st.write(
                "The Supabase view `analytics_public_summary` was not found. "
                "Run the SQL in `supabase_schema.sql` from the Supabase SQL Editor, "
                "then reboot the Streamlit app."
            )
        else:
            st.write(f"Supabase returned: {exc}")

        return

    if missing_columns:
        st.warning("analytics.csv uses an older format and cannot be summarized until it is reset or migrated.")

        if st.button("Reset analytics.csv", type="primary"):
            ANALYTICS_PATH.unlink()
            st.success("analytics.csv was reset.")
            st.rerun()

        return

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Labeled samples", summary["total_rows"])
    metric_col_2.metric("Model accuracy", format_optional(summary["model_accuracy"]))
    metric_col_3.metric("User accuracy", format_optional(summary["user_accuracy"], fallback="No guesses"))

    st.write("Class performance")
    perf_col_1, perf_col_2, perf_col_3, perf_col_4 = st.columns(4)
    perf_col_1.metric("Human precision", format_optional(summary["human_precision"]))
    perf_col_2.metric("Human recall", format_optional(summary["human_recall"]))
    perf_col_3.metric("Random precision", format_optional(summary["random_precision"]))
    perf_col_4.metric("Random recall", format_optional(summary["random_recall"]))

    st.write("Label distribution")
    st.bar_chart(label_count_frame(summary), x="label", y="rows")

    probability_df = probability_by_label_frame(summary)

    if not probability_df.empty:
        st.write("Average human probability by true label")
        st.bar_chart(probability_df, x="label", y="avg_p_human")

    st.caption(f"Guessed rows: {summary['guessed_rows']}. Raw sequences are reserved for private evaluation scripts.")


def format_optional(value, fallback="Not enough data"):
    if value is None:
        return fallback

    return f"{value:.2f}"


def is_missing_public_summary_view(exc):
    response = getattr(exc, "response", None)

    if response is not None and response.status_code == 404:
        return True

    return "404" in str(exc) and PUBLIC_SUMMARY_VIEW in str(exc)


def show_about_tab():
    st.subheader("About the experiment")
    st.write(
        "This began as a simple Human vs Random classifier. It grew into an interactive "
        "behavioral experiment: users try to fool the model, the app explains what gave "
        "them away, and real submissions are used to check whether the synthetic training "
        "data matches human behavior."
    )
    st.write(
        "The psychological idea is simple: people often know what randomness should look "
        "like, but their intuition pushes them toward sequences that feel random rather "
        "than sequences that behave randomly. The app turns that gap into something you "
        "can see, test, and measure."
    )

    st.write("Project growth")
    growth_df = pd.DataFrame(
        [
            {"Stage": "1", "Milestone": "Simple binary classifier"},
            {"Stage": "2", "Milestone": "Interpretable feature extraction"},
            {"Stage": "3", "Milestone": "Supabase real-user data logging"},
            {"Stage": "4", "Milestone": "Real-data evaluation"},
            {"Stage": "5", "Milestone": "Synthetic human generator upgrade"},
            {"Stage": "6", "Milestone": "Challenge mode and explanations"},
            {"Stage": "7", "Milestone": "Aggregate analytics and report"},
        ]
    )
    st.dataframe(growth_df, hide_index=True, width="stretch")

    st.write("What humans tend to do")
    st.caption(
        "These patterns are not mistakes in typing. They are small traces of expectation: "
        "we avoid streaks, seek balance, and add variety because those choices feel random."
    )
    bias_df = pd.DataFrame(
        [
            {
                "Human habit": "Alternation bias",
                "What it means": "Switching 0/1 too often, like 010101...",
            },
            {
                "Human habit": "Streak avoidance",
                "What it means": "Avoiding long runs such as 0000 or 1111.",
            },
            {
                "Human habit": "Balance seeking",
                "What it means": "Keeping the number of 0s and 1s too even.",
            },
            {
                "Human habit": "Repeated motifs",
                "What it means": "Reusing small chunks such as 001 or 101.",
            },
            {
                "Human habit": "Soft bit bias",
                "What it means": "Favoring one bit slightly more than the other.",
            },
        ]
    )
    st.dataframe(bias_df, hide_index=True, width="stretch")

    st.write("Synthetic human data mix")
    generator_df = pd.DataFrame(
        [
            {"Behavior": "Near-alternating", "Weight": 35},
            {"Behavior": "Balanced/streak-avoidant", "Weight": 25},
            {"Behavior": "Chunk-pattern", "Weight": 20},
            {"Behavior": "Soft-biased", "Weight": 10},
            {"Behavior": "Noisy/random-like", "Weight": 10},
        ]
    )
    st.bar_chart(generator_df, x="Behavior", y="Weight")

    st.write("Before and after the generator upgrade")
    results_df = pd.DataFrame(
        [
            {"Metric": "Synthetic accuracy", "Before": 0.785, "After": 0.880},
            {"Metric": "Synthetic ROC AUC", "Before": 0.833, "After": 0.922},
            {"Metric": "Human recall", "Before": 0.635, "After": 0.825},
            {"Metric": "Real-data accuracy", "Before": 0.889, "After": 0.899},
            {"Metric": "Real human precision", "Before": 0.825, "After": 0.850},
            {"Metric": "Real human recall", "Before": 0.863, "After": 0.863},
        ]
    )
    st.dataframe(results_df, hide_index=True, width="stretch")

    st.write("How to read the explanations")
    st.write(
        "The explanation panel is heuristic. It translates visible sequence patterns "
        "into readable signals, such as alternation bias or streak avoidance. It is meant "
        "to help users understand the behavior in their sequence, not to claim a perfect "
        "causal explanation of every model probability."
    )

    st.write("Privacy model")
    privacy_df = pd.DataFrame(
        [
            {
                "Access": "Public app",
                "Can do": "Insert labeled rows and read aggregate analytics.",
            },
            {
                "Access": "Private local scripts",
                "Can do": "Read raw Supabase rows with SUPABASE_SERVICE_ROLE_KEY.",
            },
        ]
    )
    st.dataframe(privacy_df, hide_index=True, width="stretch")

    st.info(
        "Current production model: synthetic-only Gaussian Naive Bayes with the "
        "synthetic-human-v2 data generator. Real submitted data is used for evaluation, "
        "not production retraining yet."
    )


model, scaler = load_model_assets()
ensure_session_defaults()

st.title("Human Randomness Experiment")
st.write("Try to fool a model trained to spot the patterns people leave when they imitate randomness.")

with st.sidebar:
    st.header("How To Use")
    st.write("Start with the challenge, then inspect what gave your sequence away.")
    st.write("Spaces and line breaks are ignored.")
    st.metric("Beat-the-model score", st.session_state.score)

    if supabase_enabled():
        st.success("Saving challenge data to Supabase.")
    else:
        st.info("Saving challenge data to local analytics.csv.")

tab_challenge, tab_analyze, tab_collect, tab_analytics, tab_about = st.tabs(
    ["Challenge", "Analyze", "Collect Data", "Analytics", "About"]
)

with tab_challenge:
    show_challenge_tab()

with tab_analyze:
    show_analyze_tab()

with tab_collect:
    show_collect_tab()

with tab_analytics:
    show_analytics_tab()

with tab_about:
    show_about_tab()
