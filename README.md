# Human Randomness Experiment

An interactive machine learning experiment that tests whether people can imitate randomness. Users type binary sequences, the app predicts whether each sequence is human-made or random, then explains the behavioral signals that gave the sequence away.

Live demo: https://human-random-detector-yeexavmafyev6jxjpgzpx9.streamlit.app/

For the full project narrative, evaluation results, and limitations, see [REPORT.md](REPORT.md).

## Headline Result

The current production model is **real-core hybrid trained**: real Supabase submissions are the main training signal, with capped synthetic support for coverage and class balance.

| Result | Value |
|---|---:|
| Self-collected raw Supabase rows | 468 |
| Deduplicated real evaluation rows | 455 |
| Real-data accuracy | 0.910 |
| Real-data ROC AUC | 0.970 |
| Human precision | 0.937 |
| Human recall | 0.845 |
| Random recall | 0.958 |

The real-core candidate was promoted only after it beat the previous synthetic-only baseline on the same held-out real split.

## What This Project Became

This started as a small classifier for binary strings. It grew into a portfolio-grade behavioral ML system with a deployed app, a self-collected Supabase dataset of real user submissions, real-core hybrid training, explainable predictions, aggregate analytics, calibration diagnostics, tests, and a written research report.

Figure 1: Project growth over time

| Stage | What changed |
|---:|---|
| 1 | Built a simple binary sequence classifier. |
| 2 | Added interpretable features for entropy, runs, balance, and alternation. |
| 3 | Deployed the Streamlit app and collected real user submissions in Supabase. |
| 4 | Evaluated synthetic assumptions against the self-collected dataset. |
| 5 | Improved the synthetic human generator from real-data observations. |
| 6 | Trained and promoted a real-core hybrid model with held-out real validation. |
| 7 | Added challenge mode, explanations, aggregate analytics, calibration, tests, and report. |

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
1. Real users enter sequences in the deployed app.
2. The app validates input and extracts 13 statistical features.
3. The model predicts Human or Random and explains the visible pattern signals.
4. Supabase stores the sequence, label, probabilities, metadata, and model version.
5. Private scripts split the self-collected dataset into train/test groups.
6. Real rows train the core model; capped synthetic rows provide support.
7. A promotion gate compares the candidate against the previous baseline.
8. Production artifacts update only if the held-out real metrics pass.
```

## Key Features

- Five-round "Beat the model" challenge
- One-off sequence analysis
- Plain-language explanation cards
- Synthetic human data generation based on observed human biases
- Self-collected Supabase dataset of real submitted data
- Real-core hybrid training with synthetic support capped at 1:1
- Aggregate-only public analytics
- Private real-data evaluation scripts
- Calibration diagnostics
- Real-vs-synthetic feature comparison
- Automated tests for generation, features, evaluation, analytics, and explanations

## Model And Data

The current production model is a real-core hybrid Gaussian Naive Bayes classifier. Real Supabase submissions are the main training signal, while capped synthetic data provides class-balanced support. I kept the model deliberately simple because the project is built around interpretable statistical features, not deep model complexity.

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

Figure 3: Two-step evidence path

| Step | Evidence | Outcome |
|---|---|---|
| Synthetic generator upgrade | Synthetic holdout metrics improved and real-data metrics stayed strong. | Better synthetic assumptions. |
| Real-core hybrid training | Candidate beat the previous production baseline on the same held-out real split. | Promoted production model. |

The first major upgrade improved the synthetic human generator:

| Metric | Before | After |
|---|---:|---:|
| Synthetic accuracy | 0.785 | 0.880 |
| Synthetic ROC AUC | 0.833 | 0.922 |
| Human recall | 0.635 | 0.825 |

The second major upgrade used the self-collected Supabase dataset as the core training signal. A real-core hybrid candidate was promoted after beating the synthetic-only baseline on the same held-out real split:

| Held-out real metric | Synthetic-only baseline | Real-core hybrid |
|---|---:|---:|
| Accuracy | 0.900 | 0.910 |
| ROC AUC | 0.907 | 0.945 |
| Human precision | 0.892 | 0.917 |
| Human recall | 0.846 | 0.846 |
| Macro F1 | 0.894 | 0.904 |

On the full deduplicated private real-data evaluation, the promoted model reached:

| Metric | Value |
|---|---:|
| Valid real rows | 455 |
| Accuracy | 0.910 |
| ROC AUC | 0.970 |
| Human precision | 0.937 |
| Human recall | 0.845 |
| Random recall | 0.958 |

Figure 4: Real-core promotion gate

```text
Real deployed users       -> Supabase labeled dataset
Deduped real rows         -> grouped train/test split
Real training split       -> primary model signal
Capped synthetic rows     -> support for coverage and class balance
Held-out real split       -> baseline vs candidate comparison
Promotion gate passed     -> model.pkl and scaler.pkl updated
```

Current synthetic-holdout check for the promoted production model:

| Actual \ Predicted | Random | Human |
|---|---:|---:|
| Random | 196 | 4 |
| Human | 53 | 147 |

Figure 5: Evaluation surfaces

| Evaluation surface | Purpose |
|---|---|
| Synthetic holdout report | Checks whether the promoted model still handles controlled generated examples. |
| Supabase real-data report | Checks production performance on self-collected real submissions. |
| Calibration buckets | Checks whether probabilities are trustworthy. |
| Real-vs-synthetic feature comparison | Checks whether synthetic assumptions match collected behavior. |
| Real-core training report | Saves the exact split, group IDs, weights, candidate metrics, and promotion decision. |

## Project Structure

```text
human-random-detector/
  src/
    app.py                    Streamlit app
    features.py               Feature extraction
    explanations.py           Plain-language explanation signals
    generate_data.py          Synthetic random/human generators
    real_data.py              Shared real-data cleaning/loading helpers
    train_model.py            Synthetic-only baseline training
    train_real_core_model.py  Real-core hybrid training and promotion gate
    evaluate_real_data.py     Private real-data evaluation
    analyze_real_patterns.py  Real behavior pattern analysis
    compare_synthetic_real.py Real-vs-synthetic feature comparison
  tests/                      Pytest suite
  model.pkl                   Trained model artifact
  scaler.pkl                  Trained scaler artifact
  evaluation_report.json      Synthetic evaluation report
  real_core_training_report.json Real-core promotion report
  real_data_evaluation.json   Private real-data evaluation report
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

Run the current real-core hybrid training experiment with private Supabase access:

```powershell
$env:SUPABASE_URL="https://bvgiyyynfnhusgoivdie.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="your service role key"
.\venv\Scripts\python.exe src\train_real_core_model.py
```

The older synthetic-only training script is still available for baseline regeneration:

```powershell
python src/train_model.py
```

## Run Tests

```powershell
python -m py_compile src\app.py src\features.py src\evaluate_real_data.py src\analyze_real_patterns.py src\explanations.py src\analytics_summary.py src\calibration.py src\compare_synthetic_real.py src\train_model.py src\real_data.py src\train_real_core_model.py
.\venv\Scripts\python.exe -m pytest
```

Expected result:

```text
56 passed
```

## Limitations

- The production model is now real-core hybrid, but the real dataset is still modest and should keep growing.
- Synthetic support is still used for coverage and class balance, so the project is not purely real-data trained.
- Explanation cards are heuristic descriptions of sequence patterns, not psychological diagnoses or exact causal explanations of model probabilities.
- Probabilities are reported with calibration diagnostics, but the production classifier is not recalibrated yet.

## Supabase Notes

The public app uses the anon key for inserts and aggregate analytics. Raw submitted sequences are reserved for private evaluation scripts using `SUPABASE_SERVICE_ROLE_KEY`.

To set up the database, run the full contents of [supabase_schema.sql](supabase_schema.sql) in the Supabase SQL Editor.

## What I Learned

- Synthetic data quality matters, but self-collected real data is what decides whether the assumptions hold.
- Real-data evaluation and promotion gates make the model claims much more defensible.
- Interpretable features make the app more educational and easier to debug.
- The psychology is what makes the app memorable: the model reflects back the hidden structure in a user's idea of randomness.
- Privacy changes affect both the app and the evaluation pipeline.
- A small ML model can become a much stronger project when it includes deployment, data collection, testing, and a clear research story.

## License

MIT License
