from datetime import datetime
from pathlib import Path
import secrets

import joblib
import pandas as pd
import streamlit as st

from features import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
ANALYTICS_PATH = PROJECT_ROOT / "analytics.csv"
BATCH_SIZE = 5
SEQUENCE_LENGTH = 50
REQUIRED_ANALYTICS_COLUMNS = [
    "actual_label",
    "model_prediction",
    "user_guess",
    "p_human",
]


st.set_page_config(
    page_title="Human vs Random Detector",
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
    }


def log_result(sequence, actual_label, p_human, p_random, model_prediction, user_guess):
    data = {
        "timestamp": datetime.now(),
        "sequence": sequence,
        "actual_label": actual_label,
        "p_human": p_human,
        "p_random": p_random,
        "model_prediction": model_prediction,
        "user_guess": user_guess,
    }

    df = pd.DataFrame([data])

    if ANALYTICS_PATH.exists():
        df.to_csv(ANALYTICS_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(ANALYTICS_PATH, index=False)


def load_analytics():
    if not ANALYTICS_PATH.exists():
        return None, None

    df = pd.read_csv(ANALYTICS_PATH)
    missing = [col for col in REQUIRED_ANALYTICS_COLUMNS if col not in df.columns]

    if missing:
        return df, missing

    return df, None


def show_probability_summary(result):
    st.metric("Model prediction", result["prediction"])
    st.metric("Confidence", f"{result['confidence']:.2f}")
    st.progress(result["p_human"], text=f"Human probability: {result['p_human']:.2f}")
    st.progress(result["p_random"], text=f"Random probability: {result['p_random']:.2f}")


def show_sequence_features(sequence):
    entropy, markov_entropy, kl_divergence, longest_run, alternation_rate = extract_features(sequence)

    feature_df = pd.DataFrame(
        [
            {"Feature": "Entropy", "Value": entropy},
            {"Feature": "Markov entropy", "Value": markov_entropy},
            {"Feature": "KL divergence", "Value": kl_divergence},
            {"Feature": "Longest run", "Value": longest_run},
            {"Feature": "Alternation rate", "Value": alternation_rate},
        ]
    )

    st.dataframe(feature_df, hide_index=True, use_container_width=True)


model, scaler = load_model_assets()

if "score" not in st.session_state:
    st.session_state.score = 0

for i in range(BATCH_SIZE):
    st.session_state.setdefault(f"seq_{i}", "")
    st.session_state.setdefault(f"actual_{i}", "Human")
    st.session_state.setdefault(f"guess_{i}", "Human")

st.title("Human vs Random Sequence Detector")
st.write("Analyze a bit sequence, collect labeled examples, and track how the model performs on real inputs.")

with st.sidebar:
    st.header("How To Use")
    st.write("Type a sequence of 0s and 1s. Spaces and line breaks are ignored.")
    st.write("Use Collect Data when you know the true source of the sequence.")
    st.metric("Beat-the-model score", st.session_state.score)

tab_detect, tab_collect, tab_analytics = st.tabs(
    ["Try Detector", "Collect Data", "Analytics"]
)

with tab_detect:
    st.subheader("Try one sequence")
    st.write("Paste or type a sequence to get a quick prediction without saving it to the dataset.")

    col_generate, col_clear = st.columns([1, 1])

    with col_generate:
        if st.button("Use known-random example", use_container_width=True):
            st.session_state.try_sequence = generate_random_sequence()

    with col_clear:
        if st.button("Clear", key="clear_try_sequence", use_container_width=True):
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

    if st.button("Analyze Sequence", type="primary", use_container_width=True):
        if error:
            st.error(error)
        else:
            result = predict_sequence(cleaned_sequence)
            summary_col, feature_col = st.columns([1, 2])

            with summary_col:
                show_probability_summary(result)

            with feature_col:
                st.write("Feature breakdown")
                show_sequence_features(cleaned_sequence)

with tab_collect:
    st.subheader("Collect labeled examples")
    st.write("Use this when the true source is known. Saved rows are used later for real-data evaluation.")

    action_col_1, action_col_2 = st.columns([1, 1])

    with action_col_1:
        if st.button("Generate known-random batch", use_container_width=True):
            for i in range(BATCH_SIZE):
                st.session_state[f"seq_{i}"] = generate_random_sequence()
                st.session_state[f"actual_{i}"] = "Random"
                st.session_state[f"guess_{i}"] = "Random"

    with action_col_2:
        if st.button("Clear batch", use_container_width=True):
            for i in range(BATCH_SIZE):
                st.session_state[f"seq_{i}"] = ""
                st.session_state[f"actual_{i}"] = "Human"
                st.session_state[f"guess_{i}"] = "Human"

    st.divider()

    for i in range(BATCH_SIZE):
        with st.expander(f"Example {i + 1}", expanded=i == 0):
            st.text_input(
                "Sequence",
                key=f"seq_{i}",
                placeholder="Type human data or use the random batch generator",
                help="At least 10 bits. Spaces are ignored.",
            )

            label_col, guess_col = st.columns([1, 1])

            with label_col:
                st.radio(
                    "Actual source",
                    ["Human", "Random"],
                    key=f"actual_{i}",
                    horizontal=True,
                    help="Choose Human when you typed it yourself. Choose Random for generated examples.",
                )

            with guess_col:
                st.radio(
                    "Your guess",
                    ["Human", "Random"],
                    key=f"guess_{i}",
                    horizontal=True,
                )

    if st.button("Predict And Save Valid Rows", type="primary", use_container_width=True):
        saved_rows = 0

        for i in range(BATCH_SIZE):
            sequence, error = validate_sequence(st.session_state[f"seq_{i}"])

            if error:
                st.warning(f"Example {i + 1}: {error}")
                continue

            result = predict_sequence(sequence)
            actual_label = st.session_state[f"actual_{i}"]
            user_guess = st.session_state[f"guess_{i}"]
            model_correct = result["prediction"] == actual_label
            user_correct = user_guess == actual_label

            st.write(f"Example {i + 1}: `{sequence}`")
            st.write(f"Actual: {actual_label} | Model: {result['prediction']} | Your guess: {user_guess}")
            st.write(f"Confidence: {result['confidence']:.2f}")

            if user_correct and not model_correct:
                st.success("You beat the model on this one.")
                st.session_state.score += 1
            elif user_correct and model_correct:
                st.info("You and the model were both correct.")
            elif not user_correct and model_correct:
                st.error("The model was correct.")
            else:
                st.warning("Neither prediction matched the label.")

            log_result(
                sequence=sequence,
                actual_label=actual_label,
                p_human=result["p_human"],
                p_random=result["p_random"],
                model_prediction=result["prediction"],
                user_guess=user_guess,
            )
            saved_rows += 1

        if saved_rows:
            st.success(f"Saved {saved_rows} labeled row(s) to analytics.csv.")
        else:
            st.error("No valid rows were saved.")

with tab_analytics:
    st.subheader("Collected data")

    analytics_df, missing_columns = load_analytics()

    if analytics_df is None:
        st.info("No analytics data yet. Save labeled rows from Collect Data to start tracking performance.")
    elif missing_columns:
        st.warning(
            "analytics.csv uses an older format and cannot be summarized until it is reset or migrated."
        )

        if st.button("Reset analytics.csv", type="primary"):
            ANALYTICS_PATH.unlink()
            st.success("analytics.csv was reset.")
            st.stop()
    else:
        total = len(analytics_df)
        model_correct = analytics_df["model_prediction"] == analytics_df["actual_label"]
        user_correct = analytics_df["user_guess"] == analytics_df["actual_label"]
        model_accuracy = model_correct.mean() if total else 0
        user_accuracy = user_correct.mean() if total else 0

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        metric_col_1.metric("Labeled samples", total)
        metric_col_2.metric("Model accuracy", f"{model_accuracy:.2f}")
        metric_col_3.metric("User accuracy", f"{user_accuracy:.2f}")

        st.write("Human probability over collected samples")
        st.line_chart(analytics_df["p_human"])

        with st.expander("View recent rows"):
            st.dataframe(analytics_df.tail(20), use_container_width=True, hide_index=True)
