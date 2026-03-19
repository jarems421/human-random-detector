# Human vs Random Sequence Detector

An interactive machine learning application that classifies whether binary sequences are human-generated or truly random. The model uses statistical feature engineering (entropy, Markov entropy, KL divergence, run-length patterns) combined with a Gaussian Naive Bayes classifier.

## Features
- Statistical feature extraction (entropy, transitions, structure)
- Bayesian classification model
- Multi-sequence prediction using log-likelihood aggregation
- Interactive Streamlit web app with a gamified interface
- User analytics logging for future model improvement

## How It Works
Synthetic data simulates human behaviours such as alternation bias, repetition, and noise. Each sequence is transformed into statistical features, which are used by a probabilistic model to classify sequences as human or random. Predictions across multiple inputs are combined for robustness.

## Live Demo
https://your-app-link.streamlit.app

## Project Structure
human-random-detector/
├── src/
│   ├── app.py
│   ├── main.py
│   ├── train_model.py
│   ├── features.py
│   ├── generate_data.py
├── model.pkl
├── scaler.pkl
├── requirements.txt

## Installation
git clone https://github.com/jarems421/human-random-detector.git
cd human-random-detector
pip install -r requirements.txt

## Run Locally
streamlit run src/app.py

## Key Learnings
Feature engineering is critical for model performance, probabilistic models improve interpretability, and combining multiple predictions increases robustness. Overly complex features can introduce noise.

## License
MIT License