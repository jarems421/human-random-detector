import joblib
import math
from features import extract_features

model = joblib.load("model.pkl")


def predict(sequence):
    if len(sequence) < 10:
        raise ValueError("Sequence must be at least 10 characters long")

    features = extract_features(sequence)
    probs = model.predict_proba([features])[0]

    return probs  # [P(random), P(human)]


if __name__ == "__main__":
    sequences = []

    print("Enter 5 sequences (each at least 10 characters):")

    for i in range(5):
        seq = input(f"Sequence {i+1}: ")
        sequences.append(seq)

    log_prob_random = 0
    log_prob_human = 0

    for seq in sequences:
        p_random, p_human = predict(seq)

        log_prob_random += math.log(p_random + 1e-10)
        log_prob_human += math.log(p_human + 1e-10)

    final_label = "Human" if log_prob_human > log_prob_random else "Random"

    confidence = math.exp(max(log_prob_random, log_prob_human) / len(sequences))

    print(f"\nFinal Prediction: {final_label} ({confidence:.2f})")