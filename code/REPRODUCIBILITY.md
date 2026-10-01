# Baseline single-visit analysis

## Scope

This project is derived from an earlier multi-visit project that is not part of
this repository. The present workflow uses one baseline (`BL`)
record per participant. It intentionally excludes the former multi-algorithm
benchmark stage and cross-model comparison (`step_15_*`); in this project
`step_14_*` is the bootstrap shrinkage step, not the algorithm benchmark.

The retained modeling path is:

1. select the PD cohort and one `BL` row per participant;
2. remove missing outcomes and baseline PDD records;
3. make a reproducible 70/30 stratified train/test split;
4. derive disease duration;
5. estimate all imputation values from training data only;
6. perform descriptive statistics and correlation review;
7. select features by LASSO with 10-fold stratified cross-validation; and
8. fit and evaluate one unpenalized multivariable logistic-regression model; and
9. estimate a uniform shrinkage factor by bootstrap (500 resamples of the
   training set, repeating LASSO selection and logistic refit in each), apply it
   to the step 8 coefficients with a re-estimated intercept, and evaluate the
   shrunk model on the held-out test set.

## Input

The only raw analytic input is:

`PPMI_Curated_Data_Cut_Public_20260810.xlsx`

The file is restricted row-level PPMI data and is ignored by Git. Do not
redistribute it unless the applicable PPMI terms permit redistribution.

## Environment

Run the analysis in a dedicated conda environment, never in a shared base
environment. On 2026-09-27 another project installed packages into the shared
Anaconda base mid-run; the new `array_api_compat` made scikit-learn 1.6 imports fail
in freshly spawned joblib workers and aborted a nested-CV run at fold 44 of 50.

| Environment | Lock file | torch | Use |
|---|---|---|---|
| `pdmci-tfm-gpu` | `requirements-lock-cuda.txt` | 2.14.0+cu130 | v1.4 onward |
| `pdmci-tfm` | `requirements-lock.txt` | 2.14.0 (CPU) | reproducing v1.2 only; frozen |

From the `code` directory:

```powershell
conda create -n pdmci-tfm-gpu python=3.13.5 pip -y
conda activate pdmci-tfm-gpu
python -m pip install -r requirements-lock-cuda.txt
python -m pip check
```

- `requirements-lock.txt` pins the full dependency closure (80 packages) at the
  versions that produced the v1.2 results. It deliberately excludes
  `array_api_compat`.
- `requirements-lock-cuda.txt` (97 packages) differs by the CUDA 13.0 build of
  torch, which supports the RTX 50 series (sm_120) and needs an NVIDIA driver with
  CUDA 13.0 or later (the CUDA build also runs on CPU), and by `optuna` and
  `pytabkit` with their dependencies; installing those added packages only and
  left the CV results of the existing models unchanged.
- Check on 2026-09-29 (5-fold CV of all 8 models on the 7-feature training set; the
  locked test set was not used): on CPU, `pdmci-tfm-gpu` reproduced `pdmci-tfm`
  byte for byte. On an RTX 5070 Ti, two GPU runs were byte-identical to each other;
  against CPU, out-of-fold probabilities differed by at most 0.015 (TabPFN v3.5) and
  CV AUROC by at most 0.001 (TabM). GPU CV runtimes: TabM 20-34 s (CPU 58 s),
  TabICLv2 2 s (10 s), TabPFN v3.5 12 s (32 s), TabPFN v2.5 3 s (14 s), TabFM 22 s
  (226-415 s).
- `requirements.txt` lists only the direct dependencies, for reference.

## Manual run

Preview the locked manifest without running analysis:

```powershell
python run_all.py --dry-run
```

Run the complete pipeline:

```powershell
python run_all.py
```

Resume from a numbered step after investigating a failure:

```powershell
python run_all.py --start-from 8
```

The runner stops at the first failure. Randomized operations use a fixed seed
of 42. Until a complete run finishes and its outputs are reviewed, this code is
an experimental target version rather than a validated release.

## Expected locked cohort

- Raw curated data: 20,408 rows.
- PD cohort: 8,546 visits from 1,684 participants.
- Baseline cohort: 1,684 rows from 1,684 participants.
- After missing-outcome exclusion: 1,666 participants.
- After baseline PDD exclusion: 1,665 participants.
- Training set: 1,165 participants.
- Held-out test set: 500 participants.

## Tabular Deep Learning & Foundation Model Benchmark (Steps 15-16)

The v1.4 benchmark (config 1.4.0) compares 10 algorithms with LR:

1. **LR**: the complete v1.0 algorithm (LASSO 1SE selection -> unpenalized LR ->
   bootstrap uniform shrinkage), refit inside every training fold. Fitted on the
   full training set it reproduces the v1.0 shrunk model exactly; it is the
   reference for all paired comparisons.
2. **Classical machine learning**: SVM (Platt probabilities) and random forest.
3. **GBDT**: XGBoost, LightGBM and CatBoost.
4. **Tabular deep learning trained from scratch**: RealMLP (pytabkit, NeurIPS 2024)
   and TabM (BatchEnsemble, ICLR 2025).
5. **Pretrained tabular foundation models (zero-shot)**: TabICLv2, TabPFN v3.5
   (local `v3.5` base checkpoint) and TabFM (Google Research, all training rows as
   context).

- **Tracks**: the primary analysis uses the 7 LASSO-selected features
  (`PPMI_9_*.csv`); the 28 candidate features (`PPMI_8_*.csv`) are a sensitivity
  analysis.
- **Primary metric**: test-set AUPRC, compared as the paired participant-level
  bootstrap Delta AUPRC versus LR. Delta AUROC, Delta Brier, calibration and DCA
  are secondary.
- **Device**: torch-based models run on the GPU (`protocol.device = "cuda"`).
- **Early stopping**: XGBoost, LightGBM, CatBoost, RealMLP and TabM early-stop on a
  10% split of the rows they are trained on; the rows being scored are never used
  for early stopping.
- **Class imbalance**: no model uses class weighting or resampling.
- **Failures**: a failing model stops the run.

### Hyperparameter tuning (step 15a)

`step_15_tuning.py` tunes the 7 models trained from scratch (SVM, RF, XGBoost,
LightGBM, CatBoost, RealMLP, TabM) on each track:

- **Search**: Optuna TPE (seed 42), 100 trials per model and track.
- **Objective**: mean AUPRC over the 5 training-set CV folds of the benchmark.
- **Search spaces**: pre-specified in `search_space()`.
- **TabM**: trained with 1 seed during tuning, 5 seeds in the benchmark.
- **Data**: only the training set is read.

Every finished trial is stored in `step_15_tuning_studies.db` (git-ignored), and a
rerun resumes from the completed trials. The study name carries a fingerprint of
the search space, fixed parameters, training data, CV settings, model code and
device, so a changed configuration starts a new study.

Outputs:
- `step_15_tuning_best_params.json`: best parameters, CV AUPRC and fingerprint per
  track and model.
- `step_15_tuning_trials_{track}.csv`: every trial with its parameters and CV
  metrics.

Tuning both tracks takes about 2.5 hours on an RTX 5070 Ti, mostly RealMLP.

The search space is at revision 2. In round 1, the top trials clustered at artificial
range limits, so those ranges were widened once and the GBDT tree cap was raised to
5000 (see the `search_space` docstring). All models were then re-tuned. The round-1
results are kept in `step_15_tuning_round1/`. Both rounds used the training set only.

### Benchmark (step 15b)

`step_15_benchmark.py` loads the tuned parameters. It refuses a tuned model whose
entry is missing, has fewer than 100 trials, or has a fingerprint that no longer
matches. Each finished model is checkpointed to `step_15_checkpoints/` (git-ignored
because it holds predictions), and a rerun skips models whose checkpoint
fingerprint still matches.

### Model weights

The benchmark never downloads weights for TabPFN. The v3.5 checkpoint is gated on
Hugging Face: accept the licence in a browser and download it into the TabPFN
cache (`%APPDATA%\tabpfn\`):

- `Prior-Labs/tabpfn_3_5`: `tabpfn-v3.5-20260909.safetensors`
  (SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`)
- Reproducing v1.2 also needs `Prior-Labs/tabpfn_2_5`:
  `tabpfn-v2.5-classifier-v2.5_default.ckpt`
  (SHA-256 `5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`)

The hashes match the values published on Hugging Face and are enforced through
`checkpoint_sha256` in `benchmark_config.json`. The TabPFN v3.5 and TabFM weights
carry non-commercial licences. The run records package versions, the GPU, tuned
parameters and checkpoint hashes in `step_15_benchmark_run_info.json`.

### Execution Commands

```powershell
conda activate pdmci-tfm-gpu
python step_15_tuning.py                     # step 15a, both tracks, resumable
python step_15_benchmark.py --track both     # step 15b, 7-feature track first
python step_16_benchmark_ci.py               # step 16

# Smoke tests that never touch the locked test set:
python step_15_tuning.py --track 7feat --n-trials 2 --out-dir <scratch>
python step_15_benchmark.py --quick-test --track 7feat --tuned-params <scratch>\step_15_tuning_best_params.json
```

Step 16 (about one minute) reads the saved step 15 predictions and thresholds and
refits nothing. It writes:
- per-model test-set bootstrap 95% CIs;
- forest plots of AUPRC and AUROC;
- a recalibration sensitivity analysis, in which every model is logistically
  recalibrated on its own training out-of-fold predictions. This analysis was post
  hoc in v1.2 and is pre-specified from v1.4 on.

Steps 15 and 16 are run manually and are not part of `run_all.py`. v1.2 results are
summarised in the project notes, which are not part of this repository.

`--quick-test` shrinks all compute budgets and evaluates on a stratified 30%
split of the training set; it never uses the locked test set and writes
`*_quicktest*` outputs. Known protocol deviations are listed in
`benchmark_config.json` under `protocol.known_deviations`.
