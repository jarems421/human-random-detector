import streamlit as st
import joblib
import math
import pandas as pd
import os
from datetime import datetime
from features import extract_features

# Load model + scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Beat the Model", page_icon="🧠")

st.title("🧠 Beat the Model")

st.markdown("""
Try to trick the AI into thinking your sequence is random (or human).

**Goal:** Fool the model and score points.
""")

# Initialize score
if "score" not in st.session_state:
    st.session_state.score = 0

st.write(f"### 🏆 Score: {st.session_state.score}")

st.divider()

# Input
st.subheader("Enter 5 sequences")

sequences = []

for i in range(5):
    seq = st.text_input(f"Sequence {i+1}", key=i, placeholder="e.g. 0101010011")
    sequences.append(seq)

st.divider()


# Logging function
def log_result(sequence, p_human, p_random, prediction, actual):
    file = "analytics.csv"

    data = {
        "timestamp": datetime.now(),
        "sequence": sequence,
        "p_human": p_human,
        "p_random": p_random,
        "prediction": prediction,
        "actual": actual
    }

    df = pd.DataFrame([data])

    if os.path.exists(file):
        df.to_csv(file, mode='a', header=False, index=False)
    else:
        df.to_csv(file, index=False)


# Predict button
if st.button("Predict"):
    log_prob_random = 0
    log_prob_human = 0
    valid_count = 0

    st.subheader("Results")

    for i, seq in enumerate(sequences):

        if len(seq) < 10:
            st.warning(f"Sequence {i+1} too short")
            continue

        if not all(c in "01" for c in seq):
            st.error(f"Sequence {i+1} invalid (only 0s and 1s)")
            continue

        features = extract_features(seq)
        features = scaler.transform([features])

        probs = model.predict_proba(features)[0]
        p_random, p_human = probs

        prediction = "Human" if p_human > p_random else "Random"

        st.write(f"### Sequence {i+1}")
        st.progress(float(p_human))

        col1, col2 = st.columns(2)
        col1.metric("Human", f"{p_human:.2f}")
        col2.metric("Random", f"{p_random:.2f}")

        # USER FEEDBACK (GAME PART)
        actual = st.radio(
            f"What were you aiming for? (Sequence {i+1})",
            ["Human", "Random"],
            key=f"feedback_{i}"
        )

        # Score logic
        if actual != prediction:
            st.success("You fooled the model! 🎉")
            st.session_state.score += 1
        else:
            st.error("Model got it right 😈")

        # Log result
        log_result(seq, p_human, p_random, prediction, actual)

        log_prob_random += math.log(p_random + 1e-10)
        log_prob_human += math.log(p_human + 1e-10)
        valid_count += 1

        st.divider()

    if valid_count > 0:
        final_label = "Human" if log_prob_human > log_prob_random else "Random"
        confidence = math.exp(max(log_prob_random, log_prob_human) / valid_count)

        st.subheader("Final Prediction")

        if final_label == "Human":
            st.success(f"🧠 Human ({confidence:.2f})")
        else:
            st.error(f"🎲 Random ({confidence:.2f})")

    else:
        st.error("No valid sequences entered")


# Analytics section
st.divider()
st.subheader("📊 Analytics")

if os.path.exists("analytics.csv"):
    df = pd.read_csv("analytics.csv")

    st.write("Recent activity:")
    st.dataframe(df.tail(10))

    st.write("Average human probability:")
    st.write(df["p_human"].mean())

    st.line_chart(df["p_human"])
else:
    st.write("No data yet")