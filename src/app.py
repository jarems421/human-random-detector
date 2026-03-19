import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime
from features import extract_features

# Load model + scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Human vs Random Detector")

st.title("Human vs Random Sequence Detector")
st.write("Try to predict whether a sequence is human-generated or random.")

# Score system
if "score" not in st.session_state:
    st.session_state.score = 0

st.write(f"Score: {st.session_state.score}")

st.divider()

sequences = []
guesses = []

st.subheader("Enter sequences and make your prediction")

# Input section
for i in range(5):
    seq = st.text_input(f"Sequence {i+1}", key=f"seq_{i}")
    guess = st.radio(
        f"Your guess for Sequence {i+1}",
        ["Human", "Random"],
        key=f"guess_{i}"
    )

    sequences.append(seq)
    guesses.append(guess)

st.divider()

# Logging function
def log_result(sequence, p_human, p_random, model_prediction, user_guess):
    file = "analytics.csv"

    data = {
        "timestamp": datetime.now(),
        "sequence": sequence,
        "p_human": p_human,
        "p_random": p_random,
        "model_prediction": model_prediction,
        "user_guess": user_guess
    }

    df = pd.DataFrame([data])

    if os.path.exists(file):
        df.to_csv(file, mode='a', header=False, index=False)
    else:
        df.to_csv(file, index=False)

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
        st.write(f"Model prediction: {model_prediction}")
        st.write(f"Confidence: {max(p_human, p_random):.2f}")

        # Game scoring
        if guesses[i] != model_prediction:
            st.success("You beat the model")
            st.session_state.score += 1
        else:
            st.error("Model was correct")

        # Log result
        log_result(seq, p_human, p_random, model_prediction, guesses[i])

        st.divider()

# Analytics section
st.divider()
st.subheader("Model Analytics")

try:
    if os.path.exists("analytics.csv"):
        df = pd.read_csv("analytics.csv")

        # Check correct schema
        required_cols = ["model_prediction", "user_guess", "p_human"]

        if not all(col in df.columns for col in required_cols):
            st.write("Old analytics format detected. Resetting data...")
            os.remove("analytics.csv")
            st.stop()

        total = len(df)
        correct = sum(df["model_prediction"] == df["user_guess"])
        accuracy = correct / total if total > 0 else 0

        st.metric("Model Accuracy (based on user guesses)", f"{accuracy:.2f}")
        st.metric("Total Samples", total)

        st.line_chart(df["p_human"])

    else:
        st.write("No analytics data yet")

except Exception as e:
    st.write("Analytics temporarily unavailable")