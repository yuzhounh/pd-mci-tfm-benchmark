# Baseline diagnostic model and complete manuscript workflow

## Scope

This project is derived from an earlier multi-visit project that is not part of
this repository. The present workflow uses one baseline (`BL`)
record per participant. `run_all.py` covers steps 01--14 only; the manual benchmark,
interpretability and manuscript stages below are also part of the current project.
Step 14 is bootstrap shrinkage, and step 21 supplies the authoritative corrected
seven-predictor comparison. The data cut, cohort, probabilities and selected model
remain unchanged during the v1.9 consistency/reporting revision.

The test split was used in earlier versions before some protocol choices changed.
The final tuning/fitting uses training data only, but this does not remove adaptive
bias from test reuse. The main seven-predictor/AUPRC choices were fixed for the final
benchmark after earlier results had been seen. Read `protocol.known_deviations` and
the supplement's analysis history; comparisons remain exploratory.

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

Run the baseline workflow (steps 01--14):

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

1. **LR**: step 14 selects from 28 candidates, refits unpenalized LR, estimates
   shrinkage using 500 resamples that repeat selection, and re-estimates the
   intercept. Step 21 uses this model for final seven-predictor test comparisons.
   Its five-fold OOF fits keep the selected predictors and factor 0.814 fixed;
   selection is not repeated within those folds. The legacy step 15 seven-feature
   LR used a different shrinkage factor and is superseded for the manuscript.
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
- **Internal validation**: XGBoost, LightGBM, CatBoost and TabM early-stop on a
  stratified 10% split (seed 42); RealMLP chooses its best validation epoch during
  a fixed schedule. Final fits use 1,048 rows for trees/weights and 117 for
  validation, with no subsequent all-row refit. LR/SVM/RF and foundation models
  use all 1,165 rows. The scored fold/test rows never guide these choices.
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
  hoc in v1.2 and was retained before the final v1.4 run, after earlier test
  results had been seen. Step 21 supersedes its seven-feature LR results.

Steps 15--22 are manual and are not part of `run_all.py`.

`--quick-test` shrinks all compute budgets and evaluates on a stratified 30%
split of the training set; it never uses the locked test set and writes
`*_quicktest*` outputs. Known protocol deviations are listed in
`benchmark_config.json` under `protocol.known_deviations`.

## Complete execution order and authoritative outputs

Run from `code/`, with the restricted input and licensed pretrained weights
available. R 4.6.1 with `rms` 8.1.1 generates the nomogram; `latexmk` and a TeX
distribution are required for this multi-file manuscript. Long training/tuning
and TabFM SHAP runs are resumable; the commands below describe reproduction,
not a requirement to rerun existing complete experiments for reporting edits.

```powershell
python run_all.py                         # 01--14; data, selection, final LR
python step_15_tuning.py                  # both tracks, 100 trials each
python step_15_benchmark.py --track both
python step_16_benchmark_ci.py
python step_17_shap.py                    # ten models, 100-row background, log-odds
python step_17_shap.py --models TabFM      # completes all eleven; slow GPU run
python validate_shap.py                   # targeted diagnostics and S11 checks
Rscript step_18_nomogram.R
python step_21_lr_shrinkage_consistency.py # final LR comparisons/OOF/threshold/export
python step_19_manuscript_figures.py
python step_20_manuscript_tables.py
python step_22_supplementary_tables.py
python report_cohort_metadata.py
python build_manuscript.py                # both PDFs saved under manuscript/
python validate_final_artifacts.py        # saved-result/manifest verification
```

### Display-only revisions from saved results

`feature_labels.csv` is the manuscript display dictionary for the 28 candidates;
`feature_labels.py`, the SHAP/table scripts and the R nomogram labels consume it.
Analysis columns, historical result keys and model parameter JSON stay unchanged.

For a feature-name revision, run from `code/`:

```powershell
python step_17_shap.py --plots-only
python step_20_manuscript_tables.py
python step_22_supplementary_tables.py --feature-tables-only
python refresh_saved_feature_labels.py  # Python with PyMuPDF; no R fit or LASSO CV
```

The plots-only path does not import foundation-model implementations or refit
models. It preserves the saved SHAP diagnostic CSV. The feature-tables-only path
writes only S3, S8 and S12 and skips tuning-space imports, figures and LASSO.
Copy the regenerated step 17 figures to their corresponding names under
`manuscript/figures/`; export the updated `fig1_workflow.svg` with Inkscape, then
run `build_manuscript.py` for both documents.

For label-only revisions across the workflow, nomogram, LASSO and SHAP figures,
run `python refresh_saved_feature_labels.py --display-labels-only`. It updates
saved vector text while preserving curves and plotted values; the shorter UPSIT
axis labels keep their original left/right/center alignment. UPSIT means the
age- and sex-adjusted percentile, as stated in the manuscript and captions.
Table exporters use `table_presentation.py` to keep short titles above tables,
and 9-pt notes below them, with upright bold Note and Abbreviations labels.

`refresh_saved_feature_labels.py` re-exports the saved vector nomogram and LASSO
figure with updated text, verifies that non-text vector geometry is unchanged,
and retains the nomogram's saved points. The original LASSO CV means/standard
errors are not saved as a separate file, so its existing curves, selected
penalties and numerical annotations are preserved. PyMuPDF is an optional
dependency for this display-only route (available in the local base Python);
it is not required for the scientific analysis pipeline. The default step 22
figure path still recalculates LASSO and should not be used for label-only edits.

For the current Figure 2 layout, run
`python refresh_saved_feature_labels.py --nomogram-layout-only` after any R
nomogram export. This removes the Education unit from the figure (retained in
its caption), aligns all eleven row labels on one left boundary, then moves
the entire label column 30 pt closer to the axes and trims the newly empty
left margin (670-pt cropped canvas). Text operators, fonts, scales and vector
paths are preserved. The spacing pass is idempotent. Table 1 uses bare variable names and values, with units, percentages,
reported categories and predictor definitions in the note below the table.

| Final artifact | Authority and dependencies |
|---|---|
| Final probability equation and shrinkage | `step_14_shrunk_coefficients.csv`; bootstrap repeats selection from 28 candidates |
| Seven-feature metrics, OOF threshold, CIs, paired differences, recalibration | `step_21_benchmark_*_7feat.*`; supersedes seven-feature step 15/16 outputs |
| 28-feature sensitivity comparison | `step_15_benchmark_*_full.*` and `step_16_benchmark_*_full.*` |
| Deployable JSON | `model_parameters.json["final_model"]` and the identical standalone `final_model.json`, written by `export_final_model.py`; raw units, imputation, coding and step 21 threshold |
| SHAP | `step_17_shap_*_7feat.*`; empirical training background, all 128 coalitions, all eleven models; numerical caveats in S11 |
| Main tables and updated forest/calibration figure | Steps 20 and 19 read the authorities above |
| Supplement | Step 22; printed Table S11 (file `S12.tex`) reads existing 500-bootstrap selection frequencies. Since v1.11 the TRIPOD+AI checklist is a separate document (`manuscript/tripod_ai_checklist.tex`), so printed supplementary table numbers are one lower than the `S*.tex` file names |
| Final PDFs | `build_manuscript.py`; `manuscript/manuscript.pdf` and `manuscript/supplementary_materials.pdf` beside their TeX sources; shared `manuscript/figures/` and `manuscript/tables/`; temporary builds in `tmp/latex/`, input/PDF hashes in `manuscript_build_manifest.json` |
| Current provenance and validation | `final_artifact_manifest.json`, produced by `validate_final_artifacts.py` |

Top-level coefficient fields and `shrinkage` in `model_parameters.json` are kept
as legacy development metadata, including their old ten-fold threshold. Use only
`final_model` for new predictions. Its coefficients use original units and need
no z-score transformation; education is capped at 20 years. The final threshold
is 0.2003183883873819 from pooled five-fold OOF predictions. `predict_final_model`
accepts the seven raw field names, checks binary coding, and fills missing values
with the saved training constants; missing fields or non-numeric values raise
errors. Input plausibility and multiple missing predictors require clinical review.
The saved training constants define the missing-input rule; they do not ensure
movement toward the population average risk. Performance with multiple missing
predictors has not been evaluated.

LASSO uses `solver="liblinear"`, L1, `C=1/lambda`, default
`intercept_scaling=1` (a penalized synthetic intercept), and in-fold scaling.
The 1SE rule uses `np.std(fold_scores, ddof=0)/sqrt(10)`, selecting the largest
eligible lambda. These implementation conventions were preserved. Unshrunk
association intervals are conditional on selection. Imputation precedes CV;
QUIP, ESS and RBDSQ use mode despite being numeric scores. Other numeric
predictors use medians and binary predictors modes. Non-APOE missingness is at
most 8.2%, versus APOE 19.1%/20.0% in train/test. Standardization uses the outer
training fold or all training rows; TabM tuning uses seed 42, evaluation seeds
42--46. Imputation, fixed selection and reuse of tuning folds make training CV
metrics optimistic.

## Refreshing reports from existing results

```powershell
python export_final_model.py
python report_cohort_metadata.py
python step_19_manuscript_figures.py
python step_20_manuscript_tables.py
python step_22_supplementary_tables.py --skip-lasso
python build_manuscript.py
python validate_final_artifacts.py
```

The default step 22 also recomputes the LASSO path as a consistency check;
`--skip-lasso` refreshes tables and the other figures using saved results.
Use `--tables-only` when only tables need refreshing; it leaves all figures in place.
`refresh_comparison_intervals.py` deterministically restores full-precision CI
columns in existing comparison CSVs (2,000 paired resamples, seed 42), checking
that point estimates and P values do not change. New benchmark runs now write
these columns directly. `validate_shap.py` refits only LightGBM and performs
targeted coalition/batch probes; it does not rerun full SHAP. Its checked
diagnostics are saved as `step_17_shap_diagnostics.json`. The LightGBM float32
conversion applies only to the TreeSHAP reference explainer; casting actual
prediction/coalition inputs can change probabilities and is not done.

The public export contains analysis code/configuration and the final model
parameters (`model_parameters.json`, `final_model.json`); it excludes other
results, figures, manuscript, review notes and all participant-level data. This
private workspace retains aggregate results and current PDFs.
`release/public_files.json` lists the published files and `release/export_public.py`
copies them into the local GitHub clone `release/pd-mci-tfm-benchmark/`
(`release/export_release.py` is the earlier, tag-based export). The full build/report commands require the
private manuscript sources or an equivalent separately supplied manuscript tree.
