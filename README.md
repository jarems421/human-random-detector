# Human Randomness Experiment

An interactive machine learning experiment that tests whether people can imitate randomness. Users type binary sequences, the app predicts whether each sequence is human-made or random, then explains the behavioral signals that gave the sequence away.

Live demo: https://human-random-detector-yeexavmafyev6jxjpgzpx9.streamlit.app/

For the full project narrative, evaluation results, and limitations, see [REPORT.md](REPORT.md).

## What This Project Became

This started as a small classifier for binary strings. It grew into a portfolio-grade behavioral ML system with a deployed app, a self-collected Supabase dataset of real user submissions, explainable predictions, real-data evaluation, aggregate analytics, calibration diagnostics, tests, and a written research report.

Figure 1: Project growth over time

```text
Simple classifier
  -> Feature engineering
  -> Synthetic human generator
  -> Deployed app
  -> Own Supabase dataset
  -> Real-data evaluation
  -> Improved synthetic generator
  -> Challenge mode and explanations
  -> Aggregate analytics and report
```

## Why It Is Interesting

People often make "random" sequences too tidy. They avoid long streaks, alternate too often, keep 0s and 1s too balanced, and reuse short motifs. This project turns those habits into measurable features and a deployed experiment.

The psychological angle is the part that makes the project more than a classifier. The app exposes a small cognitive bias in real time: people know randomness should be messy, but when they try to produce it themselves, they often create patterns that feel random rather than patterns that behave randomly. In that sense, the project is a tiny experiment in human intuition, pattern perception, and control.

The app does not just say "Human" or "Random." It also tells the user why:

- alternation bias
- streak avoidance
- balance seeking
- repeated motif
- soft bit bias
- random-like/noisy behavior

## How The System Works

Figure 2: App and evaluation loop

```text
Real users enter sequences
  -> validation
  -> 13-feature extraction
  -> Gaussian Naive Bayes prediction
  -> explanation signals
  -> challenge feedback
  -> Supabase logging
  -> self-collected labeled dataset
  -> private real-data evaluation
  -> synthetic generator tuning
```

## Key Features

- Five-round "Beat the model" challenge
- One-off sequence analysis
- Plain-language explanation cards
- Synthetic human data generation based on observed human biases
- Self-collected Supabase dataset of real submitted data
- Aggregate-only public analytics
- Private real-data evaluation scripts
- Calibration diagnostics
- Real-vs-synthetic feature comparison
- Automated tests for generation, features, evaluation, analytics, and explanations

## Model And Data

The current production model is a Gaussian Naive Bayes classifier trained on synthetic random and synthetic human-like sequences. I kept this model deliberately simple because the project is built around interpretable statistical features, not deep model complexity.

### Feature Contract

`extract_features()` returns 13 values in the exact order defined by `FEATURE_NAMES`. The tests assert that the feature names, feature dictionary, and extracted vector stay in sync.

| # | Feature |
|---:|---|
| 1 | entropy |
| 2 | markov_entropy |
| 3 | kl_divergence |
| 4 | longest_run |
| 5 | alternation_rate |
| 6 | balance_deviation |
| 7 | lag1_autocorrelation |
| 8 | run_count |
| 9 | mean_run_length |
| 10 | alternation_deviation |
| 11 | longest_alternating_run |
| 12 | near_alternation_score |
| 13 | pattern_break_rate |

The random class is generated with true random bit selection. The human class is generated from weighted behaviors:

| Human behavior | Weight |
|---|---:|
| Near-alternating | 35% |
| Balanced and streak-avoidant | 25% |
| Chunk-pattern | 20% |
| Soft-biased | 10% |
| Noisy/random-like | 10% |

## Results

The synthetic generator upgrade improved holdout performance:

| Metric | Before | After |
|---|---:|---:|
| Synthetic accuracy | 0.785 | 0.880 |
| Synthetic ROC AUC | 0.833 | 0.922 |
| Human recall | 0.635 | 0.825 |

The stronger check was real app data. Against 378 labeled Supabase rows, the upgraded model stayed strong:

| Metric | Old baseline | Upgraded model |
|---|---:|---:|
| Real-data accuracy | 0.889 | 0.899 |
| Real-data ROC AUC | 0.958 | 0.944 |
| Human precision | 0.825 | 0.850 |
| Human recall | 0.863 | 0.863 |
| Random recall | 0.903 | 0.919 |

The model kept human recall stable while reducing false human predictions on random sequences.

Figure 3: Synthetic assumptions checked against self-collected data

```text
Synthetic training data -> initial model
Real deployed users     -> Supabase labeled dataset
Supabase dataset        -> real-data evaluation
Real-data evaluation    -> generator upgrade
Generator upgrade       -> retrained production artifacts
```

Current synthetic confusion matrix:

| Actual \ Predicted | Random | Human |
|---|---:|---:|
| Random | 187 | 13 |
| Human | 35 | 165 |

Figure 4: Evaluation surfaces

| Evaluation surface | Purpose |
|---|---|
| Synthetic holdout report | Checks whether the model learned the controlled training task. |
| Supabase real-data report | Checks whether the model transfers to real user submissions. |
| Calibration buckets | Checks whether probabilities are trustworthy. |
| Real-vs-synthetic feature comparison | Checks whether synthetic assumptions match collected behavior. |

## Project Structure

```text
human-random-detector/
  src/
    app.py                    Streamlit app
    features.py               Feature extraction
    explanations.py           Plain-language explanation signals
    generate_data.py          Synthetic random/human generators
    train_model.py            Training and synthetic evaluation
    evaluate_real_data.py     Private real-data evaluation
    analyze_real_patterns.py  Real behavior pattern analysis
    compare_synthetic_real.py Real-vs-synthetic feature comparison
  tests/                      Pytest suite
  model.pkl                   Trained model artifact
  scaler.pkl                  Trained scaler artifact
  evaluation_report.json      Synthetic evaluation report
  supabase_schema.sql         Supabase schema and aggregate view
  REPORT.md                   Full write-up
```

## Run Locally

```powershell
git clone https://github.com/jarems421/human-random-detector.git
cd human-random-detector
pip install -r src/requirements.txt
streamlit run src/app.py
```

## Reproduce Training

```powershell
python src/train_model.py
```

## Run Tests

```powershell
python -m py_compile src\app.py src\features.py src\evaluate_real_data.py src\analyze_real_patterns.py src\explanations.py src\analytics_summary.py src\calibration.py src\compare_synthetic_real.py src\train_model.py
.\venv\Scripts\python.exe -m pytest
```

Expected result:

```text
56 passed
```

## Limitations

- The production model is trained on synthetic data; real submitted data is currently used for evaluation and generator tuning rather than production retraining.
- Real-data evaluation depends on having enough labeled Supabase rows, so its strength improves as more people use the app.
- Explanation cards are heuristic descriptions of sequence patterns, not psychological diagnoses or exact causal explanations of model probabilities.
- Probabilities are reported with calibration diagnostics, but the production classifier is not recalibrated yet.

## Supabase Notes

The public app uses the anon key for inserts and aggregate analytics. Raw submitted sequences are reserved for private evaluation scripts using `SUPABASE_SERVICE_ROLE_KEY`.

To set up the database, run the full contents of [supabase_schema.sql](supabase_schema.sql) in the Supabase SQL Editor.

## What I Learned

- Synthetic data quality matters as much as model choice.
- Real-data evaluation is the only reliable way to judge whether a synthetic generator matches human behavior.
- Interpretable features make the app more educational and easier to debug.
- The psychology is what makes the app memorable: the model reflects back the hidden structure in a user's idea of randomness.
- Privacy changes affect both the app and the evaluation pipeline.
- A small ML model can become a much stronger project when it includes deployment, data collection, testing, and a clear research story.

## License

MIT License
