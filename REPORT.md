# Human Randomness Experiment

## Problem

People are not very good at imitating randomness. When asked to type a random-looking string of 0s and 1s, they often balance the two symbols too carefully, avoid long streaks, and alternate more than true randomness would.

This project turns that observation into an interactive machine learning experiment. A user enters binary sequences, the app predicts whether the sequence is human-made or random, and the app explains which behavioral signals may have given the sequence away.

## Hypothesis

Human-made binary sequences contain measurable structure that separates them from true random sequences. The most useful signals are expected to include:

- over-alternation
- streak avoidance
- balance seeking
- short repeated motifs
- mild bit bias

The model should be judged against both synthetic holdout data and real submitted app data, because a synthetic-only score can look strong even when the generator does not match real human behavior.

## Feature Engineering

Each sequence is converted into statistical features before classification. The current feature set includes entropy, Markov transition entropy, KL divergence from a balanced distribution, longest run, alternation rate, balance deviation, lag-1 autocorrelation, run count, mean run length, alternation deviation, longest alternating run, near-alternation score, and pattern break rate.

These features are intentionally interpretable. They support both model prediction and plain-language explanations such as “you switched too often” or “you avoided long streaks.”

## Synthetic Data Upgrade

The original human generator chose broad behaviors uniformly and gave too much weight to noisy true-random-looking examples. The upgraded generator uses weighted human-like behaviors:

- 35% near-alternating
- 25% balanced and streak-avoidant
- 20% chunk-pattern
- 10% soft-biased
- 10% noisy

The random class still uses true random generation.

## Evaluation Results

After the synthetic generator upgrade, synthetic holdout performance improved:

| Metric | Before | After |
|---|---:|---:|
| Synthetic accuracy | 0.785 | 0.880 |
| Synthetic ROC AUC | 0.833 | 0.922 |
| Human precision | 0.907 | 0.927 |
| Human recall | 0.635 | 0.825 |
| Random precision | 0.719 | 0.842 |
| Random recall | 0.935 | 0.935 |

The more important check is real submitted data. Against 378 real labeled rows, the upgraded model stayed strong:

| Metric | Old baseline | Upgraded model |
|---|---:|---:|
| Real-data accuracy | 0.889 | 0.899 |
| Real-data ROC AUC | 0.958 | 0.944 |
| Human precision | 0.825 | 0.850 |
| Human recall | 0.863 | 0.863 |
| Random precision | 0.925 | 0.927 |
| Random recall | 0.903 | 0.919 |

The model kept human recall stable while improving precision and random recall. That suggests the new generator is a better fit for the real behavior collected so far.

## Calibration

The project now reports calibration diagnostics alongside accuracy and ROC AUC. Calibration checks whether predicted probabilities behave like probabilities. For example, if the model says “80% human,” roughly 80% of those examples should actually be human.

The current production model is not automatically calibrated. Calibration diagnostics are reported first so a future change can be justified by evidence rather than by adding complexity too early.

## Privacy And Analytics

The app is moving toward aggregate-only public analytics. Public visitors should see counts, rates, and charts, but not raw submitted sequences or session identifiers. Raw Supabase reads are reserved for local evaluation scripts using `SUPABASE_SERVICE_ROLE_KEY`.

This keeps the deployed demo useful while reducing unnecessary exposure of user-submitted data.

## Limitations

- The model is still trained on synthetic data only.
- The explanation layer is heuristic; it explains sequence patterns, not exact causal model internals.
- Real data volume is still modest.
- Very short sequences are inherently noisy and harder to classify reliably.
- User submissions may be biased by the app’s instructions and previous feedback.

## Future Work

- Use real submitted data in a carefully separated training/evaluation pipeline.
- Add stronger calibration if diagnostics show overconfidence.
- Track model versions across deployments.
- Expand the behavioral analysis beyond binary sequences.
- Publish a richer dashboard comparing real and synthetic feature distributions over time.
