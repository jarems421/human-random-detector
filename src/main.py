import joblib
import math
from features import extract_features


def run_training():
    print("\nTraining model...\n")
    from train_model import train_and_save_model

    train_and_save_model()


def run_prediction():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    log_prob_random = 0
    log_prob_human = 0
    valid_count = 0

    for i in range(5):
        seq = input(f"Sequence {i+1}: ").strip()

        while not all(c in "01" for c in seq):
            print("Invalid input, only 0s and 1s allowed")
            seq = input(f"Sequence {i+1}: ").strip()

        if len(seq) < 10:
            print("Too short, skipping")
            continue

        features = extract_features(seq)
        features = scaler.transform([features])

        probs = model.predict_proba(features)[0]
        p_random, p_human = probs

        print(f"Sequence {i+1} → Human: {p_human:.2f}, Random: {p_random:.2f}")

        log_prob_random += math.log(p_random + 1e-10)
        log_prob_human += math.log(p_human + 1e-10)
        valid_count += 1

    if valid_count == 0:
        print("No valid sequences.")
        return

    final_label = "Human" if log_prob_human > log_prob_random else "Random"
    confidence = math.exp(max(log_prob_random, log_prob_human) / valid_count)

    print(f"\nFinal Prediction: {final_label} ({confidence:.2f})")


def main():
    print("1. Train Model")
    print("2. Predict")

    choice = input("Choice: ").strip()

    if choice == "1":
        run_training()
    elif choice == "2":
        run_prediction()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
