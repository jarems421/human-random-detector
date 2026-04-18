# Human Randomness Experiment Report

## 1. Project Story

The first version of this project was a simple idea: train a model to decide whether a binary sequence came from a human or from a random generator. That was already a neat demo, but the project became much stronger once it started asking a better question:

Can we measure the specific ways people fail to imitate randomness?

The answer became the shape of the final system. The app now lets users try to fool the model, explains the behavioral patterns in their sequences, logs real submitted data, compares synthetic and real data, and reports calibration alongside ordinary accuracy metrics. A major part of the project is that the real benchmark was not downloaded from a standard dataset; it was collected through the deployed app itself. The current production model is now real-core hybrid trained: real submissions are the main training signal, and synthetic rows provide capped support.

Figure 1: How the project evolved

| Stage | What changed |
|---:|---|
| 1 | Simple binary sequence classifier |
| 2 | Feature engineering for entropy, runs, alternation, and balance |
| 3 | Deployed app where users submit real sequences |
| 4 | Supabase logging for real user sequences and labels |
| 5 | Real-data evaluation against the self-collected dataset |
| 6 | Synthetic generator reweighted from observed behavior |
| 7 | Challenge mode, explanations, aggregate analytics, calibration, report, and tests |
| 8 | Real-core hybrid model promoted after held-out real evaluation |

## 2. Problem And Hypothesis

People tend to make random-looking sequences too orderly. When asked to type 0s and 1s at random, they often:

- alternate too frequently
- avoid long streaks
- keep the counts of 0 and 1 too balanced
- repeat short motifs such as `001`, `010`, or `101`
- use a mild preference for one bit

The hypothesis is that these habits create measurable structure. A model should be able to separate human-made sequences from true random sequences using interpretable statistical features.

The psychological implication is that "randomness" is not only a mathematical property; it is also something people have intuitions about. Those intuitions are often biased toward fairness, balance, and visible variety. A long streak such as `00000` feels suspicious to many people, even though streaks are a normal part of true randomness. This app makes that mismatch visible.

That gives the project a behavioral science layer. The model is not just detecting bad randomness; it is detecting the traces of human expectation, pattern avoidance, and the desire to make disorder look deliberate.

## 3. System Overview

Figure 2: Prediction and learning loop

```text
Real app submission
  -> clean and validate input
  -> extract 13 statistical features
  -> scale features
  -> Gaussian Naive Bayes prediction
  -> plain-language explanation signals
  -> challenge feedback
  -> Supabase logging
  -> self-collected labeled dataset
  -> private real-data evaluation
  -> real-core hybrid training
  -> promotion gate on held-out real rows
  -> production artifact update
```

The deployed app has five main parts:

- Challenge: five attempts to fool the model
- Analyze: one-off sequence prediction
- Collect Data: known-source data collection
- Analytics: aggregate public metrics only
- About: project explanation inside the app

## 4. Feature Engineering

Each sequence is converted into 13 features:

| Feature | Why it matters |
|---|---|
| Entropy | Low entropy suggests an uneven symbol distribution. |
| Markov entropy | Captures transition variety between adjacent bits. |
| KL divergence | Measures deviation from a 50/50 bit distribution. |
| Longest run | Humans often avoid long streaks. |
| Alternation rate | Humans often switch too frequently. |
| Balance deviation | Humans often balance 0s and 1s too carefully. |
| Lag-1 autocorrelation | Captures dependence between neighboring bits. |
| Run count | Measures how often the sequence changes runs. |
| Mean run length | Summarizes streak structure. |
| Alternation deviation | Distance from random-like switching. |
| Longest alternating run | Detects long `0101...` structures. |
| Near-alternation score | Measures closeness to perfect alternation. |
| Pattern break rate | Counts disruptions in alternating patterns. |

The feature set is intentionally interpretable. The same measurements support both model prediction and explanation cards in the app.

## 5. Synthetic Data: Before And After

The first synthetic human generator used a small set of behaviors selected uniformly. It included too many noisy human examples that looked almost indistinguishable from true random data. That made the synthetic class less focused on the human biases seen in real Supabase data.

The important workflow was not "make synthetic data and trust it." The workflow became "train on controlled synthetic data, collect real user data, test the mismatch, improve the generator, then promote a real-core hybrid model only if it beats the baseline on held-out real rows." That self-collected evaluation loop is what made the project more rigorous.

The upgraded generator keeps the random class unchanged and reweights the human class:

Figure 3: Upgraded synthetic human mix

| Human behavior | Weight |
|---|---:|
| Near-alternating | 35% |
| Balanced/streak-avoidant | 25% |
| Chunk-pattern | 20% |
| Soft-biased | 10% |
| Noisy/random-like | 10% |

This generator better represents broad human randomness mistakes without overfitting to one person's style.

Figure 4: Own-dataset feedback loop

```text
Deployed Streamlit challenge
  -> users submit human/random sequences
  -> Supabase stores labels, predictions, probabilities, and metadata
  -> private evaluation scripts
  -> real-data metrics and pattern analysis
  -> real-core hybrid candidate training
  -> held-out real promotion gate
  -> promoted production artifacts
```

## 6. Evaluation Results

Synthetic holdout metrics improved after the generator upgrade:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Synthetic accuracy | 0.785 | 0.880 | +0.095 |
| Synthetic ROC AUC | 0.833 | 0.922 | +0.089 |
| Human precision | 0.907 | 0.927 | +0.020 |
| Human recall | 0.635 | 0.825 | +0.190 |
| Random precision | 0.719 | 0.842 | +0.123 |
| Random recall | 0.935 | 0.935 | 0.000 |

The next step was a real-core hybrid training run. The split used a fixed random seed and held out real groups for evaluation. The candidate used 355 real training rows, 354 synthetic support rows, real sample weight `3.0`, and synthetic sample weight `1.0`.

| Held-out real metric | Synthetic-only baseline | Real-core hybrid | Change |
|---|---:|---:|---:|
| Accuracy | 0.900 | 0.910 | +0.010 |
| ROC AUC | 0.907 | 0.945 | +0.038 |
| Human precision | 0.892 | 0.917 | +0.025 |
| Human recall | 0.846 | 0.846 | 0.000 |
| Macro F1 | 0.894 | 0.904 | +0.010 |

The candidate passed the promotion gate: human recall stayed stable, macro F1 improved, ROC AUC improved, and human precision improved.

On the full deduplicated private real-data evaluation, the promoted model reached:

| Metric | Value |
|---|---:|
| Valid real rows | 455 |
| Accuracy | 0.910 |
| ROC AUC | 0.970 |
| Human precision | 0.937 |
| Human recall | 0.845 |
| Random precision | 0.893 |
| Random recall | 0.958 |

The synthetic holdout remains useful as a check that the promoted model still handles generated examples: accuracy `0.858`, ROC AUC `0.919`.

## 7. Explainability Layer

The app now translates features into plain-language signals:

| Signal | Example interpretation |
|---|---|
| Alternation bias | "You switched between 0 and 1 more often than true random usually does." |
| Streak avoidance | "The longest streak is very short." |
| Balance seeking | "The number of 0s and 1s is almost perfectly balanced." |
| Repeated motif | "The sequence leans on a repeated short pattern." |
| Soft bit bias | "The sequence favors one bit more than expected." |
| Random-like | "This sequence is harder to separate from true random." |

This layer is heuristic. It explains the visible sequence patterns, not the exact internal causal path of the Naive Bayes classifier. That is a deliberate choice: the goal is to make the model educational and inspectable for users.

## 8. Analytics And Privacy

The app originally exposed recent raw rows for convenience. The upgraded version moves toward aggregate-only public analytics. Public visitors can see counts and rates, but raw submitted sequences and session identifiers are reserved for private evaluation scripts.

Figure 5: Public vs private data access

| Access path | Permissions | Purpose |
|---|---|---|
| Public app with anon key | Insert labeled rows and read aggregate summary view | Keeps the demo interactive without exposing raw submissions. |
| Local evaluation with service role key | Read raw analytics rows | Produces private real-data metrics and reports. |

This keeps the deployed app useful while reducing unnecessary exposure of user-submitted data.

## 9. Calibration

The project now reports calibration diagnostics, including Brier score and confidence buckets. Calibration asks whether the model's probabilities behave like probabilities. If the model says "80% human," then around 80% of those examples should actually be human.

The current production model is not automatically calibrated. The diagnostics are reported so future calibration can be justified by evidence rather than added by default.

## 10. Testing

The test suite covers:

- feature extraction
- synthetic data generation
- prediction validation
- real-data evaluation
- duplicate handling
- aggregate analytics summaries
- explanation signal classification
- real-pattern analysis

Current verification:

```text
56 tests passed
```

## 11. Limitations

- The production model is real-core hybrid, but the real-data sample is still modest.
- Synthetic support is still used for class balance and coverage.
- Explanations are heuristic and should be treated as readable diagnostics.
- Very short sequences are inherently noisy.
- User behavior may change after seeing feedback from the app.

## 12. Future Work

- Grow the self-collected dataset and rerun the real-core promotion gate.
- Track model versions across every logged prediction.
- Add richer charts for feature distributions over time.
- Evaluate calibration on larger real datasets.
- Explore whether the same approach works for longer sequences or non-binary randomness tasks.

## 13. Why This Is A Strong Portfolio Project

This project demonstrates more than a classifier. It shows the full loop:

- behavioral hypothesis
- synthetic data design
- feature engineering
- model training
- deployed app
- real-user data collection
- real-core hybrid training
- privacy-aware analytics
- real-data evaluation
- explanation UX
- reproducible tests and reports

That makes it a complete experiment rather than a standalone ML demo.

The psychology also gives the project a memorable hook. Many ML demos classify objects or predict labels, but this one invites the user to participate and then shows them something about their own intuition. That makes the app easier to explain, easier to demo, and more distinctive in a portfolio.
