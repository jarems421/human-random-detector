import streamlit as st
import joblib
import pandas as pd
import secrets
from datetime import datetime
from pathlib import Path

from features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
ANALYTICS_PATH = PROJECT_ROOT / "analytics.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(page_title="Human vs Random Detector")

st.title("Human vs Random Sequence Detector")
st.write("Type your own sequences or generate known-random ones, then see how the model responds.")

# Score system
if "score" not in st.session_state:
    st.session_state.score = 0

st.write(f"Score: {st.session_state.score}")

st.divider()

sequences = []
guesses = []
actual_labels = []

st.subheader("Collect labeled sequences")
st.write("For human data, type a sequence yourself and label it Human. For random data, use the generator below.")

if st.button("Generate 5 random sequences"):
    for i in range(5):
        st.session_state[f"seq_{i}"] = ''.join(secrets.choice(["0", "1"]) for _ in range(50))
        st.session_state[f"actual_{i}"] = "Random"

if st.button("Clear sequences"):
    for i in range(5):
        st.session_state[f"seq_{i}"] = ""
        st.session_state[f"actual_{i}"] = "Human"
        st.session_state[f"guess_{i}"] = "Human"

st.subheader("Label and predict")

# Input section
for i in range(5):
    seq = st.text_input(f"Sequence {i+1}", key=f"seq_{i}")
    actual_label = st.radio(
        f"Actual source for Sequence {i+1}",
        ["Human", "Random"],
        key=f"actual_{i}"
    )
    guess = st.radio(
        f"Your guess for Sequence {i+1}",
        ["Human", "Random"],
        key=f"guess_{i}"
    )

    sequences.append(seq)
    actual_labels.append(actual_label)
    guesses.append(guess)

st.divider()

# Logging function
def log_result(sequence, actual_label, p_human, p_random, model_prediction, user_guess):
    data = {
        "timestamp": datetime.now(),
        "sequence": sequence,
        "actual_label": actual_label,
        "p_human": p_human,
        "p_random": p_random,
        "model_prediction": model_prediction,
        "user_guess": user_guess
    }

    df = pd.DataFrame([data])

    if ANALYTICS_PATH.exists():
        df.to_csv(ANALYTICS_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(ANALYTICS_PATH, index=False)

# Prediction
if st.button("Predict"):
    st.subheader("Results")

    for i, seq in enumerate(sequences):

        if len(seq) < 10:
            st.warning(f"Sequence {i+1} too short")
            continue

        if not all(c in "01" for c in seq):
            st.error(f"Sequence {i+1} invalid (only 0s and 1s allowed)")
            continue

        features = extract_features(seq)
        features = scaler.transform([features])

        probs = model.predict_proba(features)[0]
        p_random, p_human = probs

        model_prediction = "Human" if p_human > p_random else "Random"

        st.write(f"Sequence {i+1}")
        st.write(f"Actual source: {actual_labels[i]}")
        st.write(f"Model prediction: {model_prediction}")
        st.write(f"Confidence: {max(p_human, p_random):.2f}")

        # Game scoring
        user_correct = guesses[i] == actual_labels[i]
        model_correct = model_prediction == actual_labels[i]

        if user_correct and not model_correct:
            st.success("You beat the model")
            st.session_state.score += 1
        elif user_correct and model_correct:
            st.info("You and the model were both correct")
        elif not user_correct and model_correct:
            st.error("Model was correct")
        else:
            st.warning("Neither prediction matched the label")

        # Log result
        log_result(
            seq,
            actual_labels[i],
            p_human,
            p_random,
            model_prediction,
            guesses[i],
        )

        st.divider()

# Analytics section
st.divider()
st.subheader("Model Analytics")

try:
    if ANALYTICS_PATH.exists():
        df = pd.read_csv(ANALYTICS_PATH)

        # Check correct schema
        required_cols = ["actual_label", "model_prediction", "user_guess", "p_human"]

        if not all(col in df.columns for col in required_cols):
            st.write("Old analytics format detected. Resetting data for real labels...")
            ANALYTICS_PATH.unlink()
            st.stop()

        total = len(df)
        model_correct = sum(df["model_prediction"] == df["actual_label"])
        user_correct = sum(df["user_guess"] == df["actual_label"])
        model_accuracy = model_correct / total if total > 0 else 0
        user_accuracy = user_correct / total if total > 0 else 0

        st.metric("Model Accuracy", f"{model_accuracy:.2f}")
        st.metric("User Accuracy", f"{user_accuracy:.2f}")
        st.metric("Labeled Samples", total)

        st.line_chart(df["p_human"])

    else:
        st.write("No analytics data yet")

except Exception as e:
    st.write("Analytics temporarily unavailable")
