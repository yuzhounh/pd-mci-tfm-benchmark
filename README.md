# pd-mci-tfm-benchmark

Analysis code for the manuscript *An Interpretable Diagnostic Prediction Model for Concurrent
Mild Cognitive Impairment in Parkinson's Disease: Comparison with Machine Learning, Deep
Learning and Tabular Foundation Models*.

Using baseline data of the Parkinson's Progression Markers Initiative (PPMI), the code develops a
logistic regression model (LASSO selection of seven predictors, bootstrap shrinkage) that
identifies concurrent mild cognitive impairment in Parkinson's disease (PD-MCI), and compares it
with ten other models (SVM, random forest, XGBoost, LightGBM, CatBoost, RealMLP, TabM, TabICL v2.0,
TabPFN v3.5, TabFM v1.0) in a single 70/30 train/test split, including exact SHAP values, a
nomogram and the manuscript figures and tables.

## Data are not included

The public code export contains **no PPMI data**: no participant-level rows, predictions, SHAP values or
model checkpoints. It also excludes summary results, figures, manuscript and review files, except the final model
parameters (`code/model_parameters.json` and `code/final_model.json`); the scripts regenerate
analysis outputs. The private local workspace retains aggregate results, manuscript sources and current final PDFs.
The PPMI data
(curated data cut of 10 August 2026) can be obtained from <https://www.ppmi-info.org/access-data-specimens/download-data>
by registered researchers who accept the PPMI Data Use Agreement. Place the file
`PPMI_Curated_Data_Cut_Public_20260810.xlsx` in `code/` to run the pipeline.

The pipeline is locked to this data cut. `code/step_01_extract_data.py` checks the cohort size
(1684 PD participants with baseline visits) and the required column names, and stops with an error if
they differ. A different PPMI cut will therefore not run as is; adapting `RAW_DATA_FILE` and the
`EXPECTED_*` constants in that script would yield a different analysis whose results are not
comparable with the published ones.

## Layout

| Path | Content |
|---|---|
| `code/` | Analysis scripts `step_01` to `step_23`, shared modules, configuration, environment lock files and the final model parameters |
| `code/benchmark_config.json` | Frozen benchmark protocol (models, tuning budget, seeds) |
| `code/requirements-lock-cuda.txt` | Exact package versions of the environment used for the results (Python 3.13.5, torch 2.14.0+cu130) |
| `code/REPRODUCIBILITY.md` | Environment notes and the order of the baseline pipeline |

In the private local workspace, `manuscript/` contains `manuscript.tex`,
`manuscript.pdf`, `supplementary_materials.tex`, `supplementary_materials.pdf` and `references.bib` side by side.
Both documents use the shared `manuscript/figures/` and `manuscript/tables/` directories;
supplementary files use `figS_*` and `S*.tex` names. LaTeX build files stay in `tmp/latex/`.

## Reproducing the analysis

1. Create a dedicated environment from `code/requirements-lock-cuda.txt` (do not use a shared
   base environment); a CUDA GPU was used for the deep-learning and foundation models.
2. Put the PPMI file in `code/` and run the baseline pipeline (data preparation, LASSO, logistic
   regression, bootstrap shrinkage) from `code/`:

   ```bash
   python run_all.py
   ```

3. Run the benchmark and the later steps in this order: `step_15_tuning.py` (hyperparameter
   tuning), `step_15_benchmark.py`, `step_16_benchmark_ci.py`, `step_17_shap.py`,
   `step_18_nomogram.R` (R with the `rms` package), `step_21_lr_shrinkage_consistency.py`,
   `step_19_manuscript_figures.py`, `step_20_manuscript_tables.py`, `step_22_supplementary_tables.py`,
   and the sensitivity analysis `step_23_lr_reduced_sensitivity.py`.
   Long steps save each finished unit (fold, model, chunk) to disk and resume after an interruption.
   `step_17_shap.py` computes exact SHAP values for all eleven models. By default it runs the ten
   models other than TabFM v1.0; TabFM v1.0 is slow (about 5 hours on an RTX 5070 Ti GPU) and is run
   separately with `python step_17_shap.py --models TabFM`, after which the figures and tables cover
   all eleven models. Run `python validate_shap.py` for the targeted LightGBM/TabICL diagnostics
   used in the final SHAP checks and artifact validation.
   Steps 19, 20 and 22 write figures and tables to `manuscript/` next to `code/`.
   With the manuscript sources available, run `python report_cohort_metadata.py`,
   `python build_manuscript.py` and `python validate_final_artifacts.py` to refresh cohort metadata,
   compile both final PDFs and verify the complete saved artifact chain. The public export does not
   supply manuscript TeX sources. See `code/REPRODUCIBILITY.md` for the complete order and authority map.

Step 21 uses the final logistic regression model (shrinkage factor from the bootstrap that repeats
LASSO selection) in the seven-predictor comparison. Its outputs supersede the legacy seven-predictor
step 15/16 results. The deployable raw-unit model, imputation values and current threshold are in
`code/model_parameters.json["final_model"]`, generated and checked by `code/export_final_model.py`, which also writes the same object alone to `code/final_model.json`.
The test split was evaluated in earlier versions before the primary track/metric and comparison
protocol changed; final comparisons are exploratory and need independent confirmation.

## Pretrained model weights

The weights of the tabular foundation models are **not** included and are subject to their own
licences. TabPFN v3.5 (`Prior-Labs/tabpfn_3_5`) and TabFM v1.0 (`google/tabfm-1.0.0-pytorch`) are
released under non-commercial licences; obtain them from their providers and check that your use is
permitted. TabICL v2.0 uses the checkpoint `tabicl-classifier-v2-20260212.ckpt`.

## Licence

The code is released under the MIT licence (see `LICENSE`). This does not apply to the PPMI data or
to the pretrained model weights.

## Citation

See `CITATION.cff`.
