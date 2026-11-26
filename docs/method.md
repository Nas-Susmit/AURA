# Implementation guide

## Pipeline and mapping to the manuscript

1. `gen` generates latent binary labels, correlated features and a mimic flag. `acquire` applies independent quality-related observation noise and modality acquisition mechanisms. Some test shifts generate independent populations. The simulator repairs all-zero masks by observing one uniformly random modality; the demo separately defers if no evidence is supplied.
2. `feat` masks unavailable raw features and appends availability and observed quality for early logistic, RF and HGB. Bootstrap modality experts use simulated observations with quality as an input; all generated features are available during expert training, even when simulated acquisition later hides them. This is an illustrative simulator design, not realistic clinical data access.
3. `gf` and `gate2` fit ridge regression to modality-specific squared error on gate records, invert clipped predicted error, and normalize across available modalities (`fuse`). Equal and quality fusion are controls.
4. `lgt` and `cal` fit/apply a logistic recalibrator on a disjoint calibration set. The probability and hard class label remain unchanged in ranking ablations.
5. `ef` constructs binary entropy, weighted ensemble variance, observed fraction, observed mean quality and masks; append four times weighted variance of expert probabilities (4D). `shared` uses equal-weight context for every classifier to allow matched comparisons. Native AURA uses gate weights for its own ablation. Separate rankers are fitted on the ranker development split. The 80th-percentile cutoff is set there; realized held-out coverage is not necessarily 80%.
6. `rc` computes expected retained error at every rank using mean errors within exact ties, and `met` averages across all retained sizes for AURC. `risk80` is risk at fixed test coverage of 80%, not the fixed-threshold risk in `thresholds.csv`.
7. `explain` computes exact coalition Shapley attribution for fixed weights and calibration, using calibrated 0.5 as reference. First-member feature contributions are logistic logit decompositions, not end-to-end SHAP. Stability compares independently trained models at fixed synthetic anchor observations.
8. Seed-level paired t intervals average across the seven conditions first. Review-motivated tests are exploratory; no equivalence, clinical significance or patient-level inference follows from them.

## Data dictionary

| Columns | Meaning |
|---|---|
| stiffness, movement_relief, night_pain, extra_articular | Synthetic binary clinical indicators |
| hla_b27 | Synthetic binary laboratory indicator |
| crp_z, esr_z | Dimensionless Gaussian proxies, not lab measurements |
| bme, erosion | Synthetic binary structured imaging indicators, not images |
| structural_score | Dimensionless synthetic proxy |
| synthetic_label | Generator class, not adjudicated disease |
| latent_mimic_audit_only | Simulator-only mimic flag; excluded from predictors |
| split | train (5000), gate (1500), calibration (1000), uncertainty (1000), test (1500) |

`acquire` generates masks and quality separately for each condition; they are not present in the clean example CSV. Scenario names: `complete`, `mcar30`, `mcar50`, `quality_shift`, `selective_acquisition`, `mimic_shift`, `combined_shift`. Simulated quality is conveniently observed; the low-quality scenario changes both quality draws and resulting noise. Selective acquisition uses latent class in the generator and is not a deployable clinical procedure.

## Output schema

- `metrics.csv`: model, ranking, seed, scenario, AUROC, AUPRC, Brier, ECE (10 equal-width bins), log loss, accuracy, AURC, risk at 80%/50% coverage, error detection AUROC.
- `paired.csv`: comparator, ranking contrast, metric, seed-level mean difference and two-sided t interval.
- `sensitivity.csv`: nine one-factor settings, paired native AURC and delta for each seed/condition.
- `thresholds.csv`: realized held-out coverage and selective risk at development-selected native cutoff.
- `curves.csv`: sampled matched full risk-coverage points (AURC uses all ranks).
- `experts.csv`: observed-case expert metrics before/after calibration.
- `coefficients.csv`, `permutation.csv`: standardized disagreement coefficients and held-out feature permutation effects.
- `explanation_stability.csv`, `feature_stability.csv`: pairwise seed attribution agreement; pairwise rows are not independent samples.

Full results use deterministic seeds and pinned package versions. Cross-platform floating point/solver differences may occur; verification will flag differences outside tolerance rather than silently pass.
