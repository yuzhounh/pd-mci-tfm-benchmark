"""Step 22: Supplementary tables and figures for the manuscript, from saved results.

Tables and calibration/decision curves use saved results. The LASSO figure reuses saved coefficient paths and existing vector-derived display curves;
--tables-only skips all figure work.

Outputs (in ../manuscript/): tables/S*.tex and figures/figS_*.pdf
Use --tables-only to refresh the tables without recomputing LASSO or figures.
Use --feature-tables-only for S3/S8/S12 only, without model/tuning imports.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from result_formatting import ci_pair, difference, number
from feature_labels import (
    FEATURE_DEFINITIONS,
    FEATURE_LABELS,
    PREDICTOR_NOTE,
    PREDICTOR_ABBREVIATIONS,
)
from table_presentation import format_table_notes
from supplementary_layout import supplementary_layout

FEATURE_TABLES_ONLY = "--feature-tables-only" in sys.argv
if not FEATURE_TABLES_ONLY:
    from step_15_tuning import search_space

OUT = os.path.join("..", "manuscript")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)

MODELS = [
    "LR",
    "SVM",
    "RF",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RealMLP",
    "TabM",
    "TabICLv2",
    "TabPFN",
    "TabFM",
]
LABEL = {"TabICLv2": "TabICL v2.0", "TabPFN": "TabPFN v3.5", "TabFM": "TabFM v1.0"}
BS = chr(92)


def name(m: str) -> str:
    return LABEL.get(m, m)


def esc(s) -> str:
    s = str(s)
    s = s.replace(BS, "@@BS@@")
    for a, b in (
        ("&", BS + "&"),
        ("%", BS + "%"),
        ("_", BS + "_"),
        ("#", BS + "#"),
        ("$", BS + "$"),
    ):
        s = s.replace(a, b)
    return s.replace("@@BS@@", BS + "textbackslash{}").replace(
        "ε", "$" + BS + "varepsilon$"
    )


def num(x, d=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}".replace("-", "$-$")


def ci_str(row, key, d=3) -> str:
    return f"{num(row[key], d)} " + ci_pair(
        num(row[key + "_CI_lower"], d), num(row[key + "_CI_upper"], d)
    )


FULL_WIDTH = {"S2.tex", "S3.tex", "S4.tex", "S5.tex", "S7.tex"}


def write_table(
    fname,
    caption,
    label,
    header,
    rows,
    colspec,
    small="footnotesize",
    longtable=False,
    resize=False,
):
    if FEATURE_TABLES_ONLY and fname not in {"S3.tex", "S8.tex", "S12.tex"}:
        return
    # Bold short title = first sentence of the caption
    if "||" in caption:
        first, rest = (x.strip() for x in caption.split("||", 1))
        caption = f"{BS}textbf{{{first}}} {rest}"
    else:
        first, sep, rest = caption.partition(". ")
        caption = f"{BS}textbf{{{first}{'.' if sep else ''}}}" + (
            " " + rest if sep else ""
        )
    if fname == "S3.tex":
        header = [f"{BS}ManuscriptHeader{{{h}}}" for h in header]
    hdr = " & ".join(header) + f" {BS}{BS}"
    # Page-width tables (all supplementary tables S2-S5 and S7 except S6; S8 and S9 are scaled to the text width)
    full = fname in FULL_WIDTH and not resize
    fill = "@{" + BS + "extracolsep{" + BS + "fill}}"
    if longtable:
        colspec = fill + colspec if full else colspec
        lines = [f"{{{BS}{small}"]
        if full:
            lines.append(
                f"{BS}setlength{{{BS}LTleft}}{{0pt}}{BS}setlength{{{BS}LTright}}{{{BS}fill}}"
            )
        lines += [
            f"{BS}begin{{longtable}}{{{colspec}}}",
            f"{BS}caption{{{caption}}}{BS}label{{{label}}}{BS}{BS}",
            f"{BS}toprule",
            hdr,
            f"{BS}midrule",
            f"{BS}endfirsthead",
            f"{BS}toprule",
            hdr,
            f"{BS}midrule",
            f"{BS}endhead",
            f"{BS}bottomrule",
            f"{BS}endfoot",
        ]
    else:
        env = "tabular*" if full else "tabular"
        begin = (
            f"{BS}begin{{tabular*}}{{{BS}textwidth}}{{{fill}{colspec}}}"
            if full
            else f"{BS}begin{{tabular}}{{{colspec}}}"
        )
        lines = [
            f"{BS}begin{{table}}[H]",
            f"{BS}centering",
            f"{BS}{small}",
            f"{BS}caption{{{caption}}}",
            f"{BS}label{{{label}}}",
            (f"{BS}resizebox{{{BS}textwidth}}{{!}}{{%" if resize else ""),
            begin,
            f"{BS}toprule",
            hdr,
            f"{BS}midrule",
        ]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + f" {BS}{BS}")
    if longtable:
        lines += [f"{BS}end{{longtable}}", "}"]
    else:
        lines += [
            f"{BS}bottomrule",
            f"{BS}end{{{env}}}" + ("}" if resize else ""),
            f"{BS}end{{table}}",
        ]
    with open(os.path.join(TAB, fname), "w", encoding="utf-8", newline="\n") as f:
        f.write(
            supplementary_layout(fname, format_table_notes("\n".join(lines) + "\n"))
        )


config = json.load(open("benchmark_config.json", encoding="utf-8"))

# ---------------------------------------------------------------- S2 outcome
le = BS + "le"
rows = [
    ("HVLT-R total recall", "T-score", f"${le} 35$", "age"),
    ("HVLT-R recognition discrimination index", "T-score", f"${le} 35$", "age"),
    (
        "Benton Judgment of Line Orientation",
        "scaled score",
        f"${le} 6$",
        "age and education",
    ),
    ("Letter--Number Sequencing", "scaled score", f"${le} 6$", "age"),
    ("Semantic (animal) fluency", "T-score", f"${le} 35$", "age and education"),
    ("Symbol Digit Modalities Test", "T-score", f"${le} 35$", "age and education"),
]
write_table(
    "S2.tex",
    "Definition of the outcome, concurrent PD-MCI. The supplied PPMI-derived indicator is based on at least two of the six normatively adjusted scores at or below the listed cut-offs. These cut-offs and the supplied indicator define the analysis outcome; they were not replaced by a strict inequality at 1.5 standard deviations. Scores were used without re-standardization. The PPMI data dictionary cites the HVLT-R Professional Manual for total recall norms and does not state a normative source for the recognition discrimination index.",
    "tab:S2",
    ["Test", "Score", "Cut-off", "Adjusted for"],
    rows,
    "llll",
)

# ---------------------------------------------------------------- S3 predictors and missingness
mapping = pd.read_csv("PPMI_feature_mapping.csv", encoding="utf-8-sig").set_index(
    "feature"
)
for feature, definition in FEATURE_DEFINITIONS.items():
    mapping.loc[feature, ["abbreviation", "full name"]] = [
        definition["label"],
        definition["description"],
    ]
NF = ["SITE", "subgroup", "EVENT_ID", "visit_date", "PATNO", "MCI"]
cand = [c for c in pd.read_csv("PPMI_8_train.csv", nrows=1).columns if c not in NF]
sel = [
    c
    for c in pd.read_csv("PPMI_9_train.csv", nrows=1).columns
    if c not in ("PATNO", "MCI")
]
EXTRA = {
    "upsit_pctl": {
        "abbreviation": "UPSIT",
        "full name": "University of Pennsylvania Smell Identification Test, percentile adjusted for age and sex",
    },
    "MIA_CAUDATE_mean": {
        "abbreviation": "Caudate SBR",
        "full name": "Mean specific binding ratio of the left and right caudate on DAT imaging",
    },
    "MIA_PUTAMEN_mean": {
        "abbreviation": "Putamen SBR",
        "full name": "Mean specific binding ratio of the left and right putamen on DAT imaging",
    },
}
tr7, te7 = pd.read_csv("PPMI_6_train.csv"), pd.read_csv("PPMI_6_test.csv")
imp_par = json.load(open("step_08_imputation_parameters.json", encoding="utf-8"))


def imp_value(c):
    # training-set median (continuous) or mode (categorical) used for missing values; original units
    p = imp_par[c]
    v = float(p["value"])
    txt = f"{v:.0f}" if v == int(v) else f"{v:.2f}"
    return txt


def missing_count(series):
    """Report missing count and percentage from the pre-imputation data."""
    count = int(series.isna().sum())
    return f"{count} ({100 * count / len(series):.1f})"


rows = []
for c in cand:
    r = mapping.loc[c] if c in mapping.index else EXTRA[c]
    rows.append(
        (
            esc(r["abbreviation"]),
            esc(r["full name"]),
            missing_count(tr7[c]),
            missing_count(te7[c]),
            imp_value(c),
            "yes" if c in sel else "",
        )
    )
write_table(
    "S3.tex",
    "The 28 candidate predictors and their missing values.|| Missing data are reported as n (\\%) before imputation (training set, n = 1,165; test set, n = 500). Imputation values, in original units, were estimated from the training set and applied unchanged to the test set. Binary and categorical predictors were imputed using the mode; numeric measures and scores were imputed using the median, except QUIP, ESS and RBDSQ, which used the mode. LASSO indicates whether the predictor was selected by LASSO. Sex: 1 = male; other binary predictors: 1 = yes. Disease duration was derived after imputing age at diagnosis and onset, so its zero missingness does not imply complete source ages.",
    "tab:S3",
    [
        "Display name",
        "Description",
        BS + "makecell{Train" + BS + BS + "n (" + BS + "%)}",
        BS + "makecell{Test" + BS + BS + "n (" + BS + "%)}",
        "Imputation",
        "LASSO",
    ],
    rows,
    "l>{" + BS + "raggedright" + BS + "arraybackslash}p{4.9cm}cccc",
    longtable=True,
)


# ---------------------------------------------------------------- S4 fixed settings
def flat(d, prefix=""):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += flat(v, prefix)
        else:
            out.append(f"{k} = {v}")
    return out


rows = []
for m in MODELS:
    c = config["models"][m]
    fixed = {
        k: v
        for k, v in c.items()
        if k
        not in (
            "display_name",
            "type",
            "checkpoint",
            "checkpoint_sha256",
            "pipeline",
            "n_shrinkage_bootstrap",
        )
    }
    txt = "; ".join(flat(fixed)) or "library defaults"
    if m == "LR":
        txt = "LASSO selection, unpenalized fit and 500-bootstrap uniform shrinkage; no tuned parameters"
    elif m in {"SVM", "RF", "XGBoost", "LightGBM", "CatBoost"}:
        txt += "; random seed = 42"
    elif m == "RealMLP":
        txt = "random seed = 42; validation metric = cross-entropy; retain best epoch during fixed schedule"
    if "checkpoint" in c:
        # Pretrained weights actually used (file name or repository, without the download notes)
        weights = c["checkpoint"].split(" (")[0]
        base = "; base model" if m == "TabPFN" else ""
        txt = f"weights = {weights}{base}" + (
            "" if txt == "library defaults" else "; " + txt
        )
    rows.append((esc(name(m)), esc(c["type"].replace("_", " ")), esc(txt)))
write_table(
    "S4.tex",
    "Model types and fixed (non-tuned) settings. Pretrained weights are given by file name or repository. Hyperparameters of tuned models are in Table~S4; other estimator settings use library defaults. The internal validation split is stratified: 117 of 1,165 participants in final fitting, leaving 1,048 for weight/tree fitting in XGBoost, LightGBM, CatBoost, RealMLP and TabM. These models were not refitted on all rows. Validation uses log-loss/cross-entropy (binary cross-entropy for TabM); TabM evaluation averages five random seeds. LR/SVM/RF and foundation-model fitting/context use all rows. The benchmark standardization uses the outer training fold or full training set; the final LR equation is reported in original units and requires unstandardized inputs.",
    "tab:S4",
    ["Model", "Type", "Fixed settings"],
    rows,
    "l>{"
    + BS
    + "raggedright"
    + BS
    + "arraybackslash}p{3cm}>{"
    + BS
    + "raggedright"
    + BS
    + "arraybackslash}p{9cm}",
    longtable=True,
)


# ---------------------------------------------------------------- S5 search spaces and selected values
class Recorder:
    def __init__(self):
        self.space = {}

    def suggest_float(self, n, low, high, log=False, step=None):
        self.space[n] = f"{low:g}--{high:g}" + (" (log)" if log else "")
        return low

    def suggest_int(self, n, low, high, step=None, log=False):
        self.space[n] = (
            f"{low}--{high}"
            + (f", step {step}" if step else "")
            + (" (log)" if log else "")
        )
        return low

    def suggest_categorical(self, n, choices):
        self.space[n] = ", ".join(str(c) for c in choices)
        return choices[-1]


best = json.load(open("step_15_tuning_best_params.json", encoding="utf-8"))
rows = []
for mi, m in enumerate([] if FEATURE_TABLES_ONLY else MODELS[1:8]):
    rec = Recorder()
    search_space(m, rec)
    for pi, (p, rng) in enumerate(rec.space.items()):
        vals = []
        for track in ("7feat", "full"):
            v = best[track][m]["params"].get(p, "--")
            vals.append(num(v, 4) if isinstance(v, float) else esc(v))
        # The model name is written once, on its first row; a small gap separates the models
        label = ""
        if pi == 0:
            label = (BS + "addlinespace " if mi > 0 else "") + esc(name(m))
        rows.append((label, esc(p), esc(rng), vals[0], vals[1]))
write_table(
    "S5.tex",
    "Hyperparameter search spaces and selected values. Each model was tuned with the tree-structured Parzen estimator (100 trials per model and predictor set); the objective was the mean AUPRC over five cross-validation folds of the training set. Parameters not listed were fixed (Table~S3).",
    "tab:S5",
    [
        "Model",
        "Parameter",
        "Search range",
        BS + "makecell[l]{Selected," + BS + BS + " 7 predictors}",
        BS + "makecell[l]{Selected," + BS + BS + " 28 predictors}",
    ],
    rows,
    "lllll",
    longtable=True,
)

# ---------------------------------------------------------------- S6 training-set CV
cv = pd.read_csv("step_21_benchmark_cv_metrics_7feat.csv").set_index("Model")
rows = []
for m in MODELS:
    r = cv.loc[m]
    rows.append(
        (esc(name(m)),)
        + tuple(
            f"{num(r[f'{k}_mean'], 3)} ({num(r[f'{k}_std'], 3)})"
            for k in ("AUPRC", "AUROC", "Brier")
        )
    )
write_table(
    "S6.tex",
    "Cross-validated performance in the training set.|| Five folds; mean and standard deviation across folds; seven predictors. For the tuned models these values are optimistic because the same folds were used to choose hyperparameters, and the seven predictors were chosen by LASSO in the full training set; they are not compared with LR. For LR, the seven predictors and the shrinkage factor of the final model (0.814) were fixed in every fold.",
    "tab:S6",
    ["Model", "AUPRC", "AUROC", "Brier score"],
    rows,
    "lccc",
)

# ---------------------------------------------------------------- S7 classification at thresholds
ci7 = pd.read_csv("step_21_benchmark_model_ci_7feat.csv").set_index("Model")
rows = []
for m in MODELS:
    r = ci7.loc[m]
    rows.append(
        (esc(name(m)), num(r["Threshold"], 3))
        + tuple(ci_str(r, k, 3) for k in ("Sensitivity", "Specificity", "PPV", "NPV"))
    )
write_table(
    "S7.tex",
    "Classification performance in the test set at each model's Youden threshold.|| The threshold is the Youden-optimal cut-off of the pooled out-of-fold predictions in the training set (seven predictors; 95% bootstrap CIs).".replace(
        "%", BS + "%"
    ),
    "tab:S7",
    ["Model", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV"],
    rows,
    "lccccc",
)

# ---------------------------------------------------------------- S8 SHAP importance
imp = pd.read_csv("step_17_shap_importance_7feat.csv")
shap_models = [m for m in MODELS if m in set(imp["Model"])]
order = list(imp[imp["Model"] == "LR"].sort_values("Rank")["Predictor"])
rows = []
for p in order:
    row = [esc(FEATURE_LABELS[p])]
    for m in shap_models:
        r = imp[(imp["Model"] == m) & (imp["Predictor"] == p)].iloc[0]
        row.append(f"{num(r['Mean_abs_SHAP'], 3)} ({int(r['Rank'])})")
    rows.append(row)
write_table(
    "S8.tex",
    "Mean absolute SHAP value of each predictor in the test set.|| Log-odds scale, with the rank within the model in parentheses; eleven models. Predictors are ordered by their rank in LR. "
    + PREDICTOR_NOTE
    + BS
    + "textit{ Abbreviations:} "
    + PREDICTOR_ABBREVIATIONS
    + " LR, logistic regression; RF, random forest; SHAP, Shapley additive explanations; SVM, support vector machine.",
    "tab:S8",
    ["Predictor"] + [esc(name(m)) for m in shap_models],
    rows,
    "l" + "c" * len(shap_models),
    small="scriptsize",
    resize=True,
)
chk = pd.read_csv("step_17_shap_checks_7feat.csv").set_index("model")
rows = []
for m in shap_models:
    r = chk.loc[m]
    rows.append(
        (
            esc(name(m)),
            f"{r['repro_max_abs_diff_prob']:.1e}",
            f"{r['additivity_max_err']:.1e}",
            (
                "--"
                if np.isnan(r["lr_linear_max_diff"])
                else f"{r['lr_linear_max_diff']:.1e}"
            ),
            (
                "--"
                if np.isnan(r["treeshap_max_diff"])
                else f"{r['treeshap_max_diff']:.1e}"
            ),
        )
    )
write_table(
    "S11.tex",
    "Checks of the SHAP computation. Refit: maximum difference between probabilities from the refitted models and the reported predictions (LR: final model). Additivity: maximum error of the base value plus SHAP sum relative to predicted log-odds. LR formula: maximum difference from $"
    + BS
    + "beta_j(x_j-"
    + BS
    + "bar{x}_j)$. TreeSHAP: difference from the interventional reference. Larger differences for LightGBM and TabICL reflected numerical precision (float64 reference inputs for LightGBM and mixed-precision batch effects for TabICL); LightGBM global ranks were unchanged with float32 reference inputs. Enumeration is exact for the stated empirical background, up to numerical precision.",
    "tab:S11",
    ["Model", "Refit (probability)", "Additivity (log-odds)", "LR formula", "TreeSHAP"],
    rows,
    "lcccc",
)

# ---------------------------------------------------------------- S9 28-predictor sensitivity
cif = pd.read_csv("step_16_benchmark_model_ci_full.csv").set_index("Model")
dff = pd.read_csv("step_15_benchmark_bootstrap_diffs_full.csv").set_index("Model")


def dfmt(d, s):
    i = s.index("–", 1)
    return f"{num(d)} " + ci_pair(num(float(s[:i])), num(float(s[i + 1 :])))


D = BS + "Delta"
rows = []
for m in MODELS:
    r = cif.loc[m]
    if m == "LR":
        da = du = db = p = "--"
    else:
        d = dff.loc[m]
        da, du = difference(d, "AUPRC"), difference(d, "AUROC")
        db = difference(d, "Brier")
        pv = d["Delta_AUPRC_pval"]
        p = f"{pv:.2f}" if pv >= 0.095 else f"{pv:.3f}"
    rows.append(
        (
            esc(name(m)),
            ci_str(r, "AUPRC"),
            da,
            p,
            ci_str(r, "AUROC"),
            du,
            num(r["Brier"], 3),
            db,
            num(r["Calib_Slope"], 2),
        )
    )
write_table(
    "S9.tex",
    f"Sensitivity analysis with all 28 candidate predictors.|| Test-set performance and paired differences from LR (model minus LR; 2,000 bootstrap resamples; $P$ for ${D}$AUPRC). LR selects its own predictors and is the same model as in the primary analysis. The seven tuned models were tuned again on the 28 predictors. No adjustment was made for multiple comparisons.",
    "tab:S9",
    [
        "Model",
        "AUPRC",
        f"${D}$AUPRC",
        "$P$",
        "AUROC",
        f"${D}$AUROC",
        "Brier",
        f"${D}$Brier",
        "Slope",
    ],
    rows,
    "lcccccccc",
    small="scriptsize",
    resize=True,
)

# ---------------------------------------------------------------- S10 recalibration
rec = pd.read_csv("step_21_benchmark_recalibrated_7feat.csv").set_index("Model")
rows = []
for m in MODELS:
    r = rec.loc[m]
    dB = (
        "--"
        if np.isnan(r["Delta_Brier_vs_LR"])
        else f"{number(r['Delta_Brier_vs_LR'], 4)} "
        + ci_pair(
            number(r["Delta_Brier_CI_lower"], 4), number(r["Delta_Brier_CI_upper"], 4)
        )
    )
    rows.append(
        (
            esc(name(m)),
            num(r["Recal_a"], 2),
            num(r["Recal_b"], 2),
            num(r["Brier"], 3),
            dB,
            ci_str(r, "Calib_Slope", 2),
        )
    )
write_table(
    "S10.tex",
    "Recalibration analysis (seven predictors). Every model, including LR, was recalibrated with logit(p') = a + b logit(p), with a and b fitted on its own out-of-fold training predictions. AUROC and AUPRC are unchanged by recalibration. "
    + f"${D}$Brier is the model minus recalibrated LR.",
    "tab:S10",
    [
        "Model",
        "$a$",
        "$b$",
        "Brier",
        f"${D}$Brier vs LR (95% CI)".replace("%", BS + "%"),
        "Calibration slope (95% CI)".replace("%", BS + "%"),
    ],
    rows,
    "lccccc",
)

# ---------------------------------------------------------------- S12 bootstrap selection stability
freq = pd.read_csv("step_14_bootstrap_selection_frequency.csv")
rows = []
for _, r in freq.iterrows():
    feature = r["Feature"]
    label = (
        mapping.loc[feature]["abbreviation"]
        if feature in mapping.index
        else EXTRA[feature]["abbreviation"]
    )
    count = int(round(500 * r["Selection_frequency"]))
    rows.append(
        (
            esc(label),
            count,
            num(100 * r["Selection_frequency"], 1),
            "yes" if r["In_final_model"] else "no",
        )
    )
write_table(
    "S12.tex",
    "Bootstrap predictor selection frequencies.|| In 500 training-set resamples, the full LASSO selection and unpenalized refit procedure was repeated from the 28 candidates. Frequencies describe selection stability, not probabilities that associations are causal. The deployable model retains the fixed seven predictors selected from the original training set. Predictor definitions and coding are in Table S2. "
    + PREDICTOR_NOTE
    + BS
    + "textit{ Abbreviations:} "
    + PREDICTOR_ABBREVIATIONS
    + " APOE, apolipoprotein E; BMI, body mass index; ESS, Epworth Sleepiness Scale; H"
    + BS
    + "&Y, Hoehn and Yahr stage; LASSO, least absolute shrinkage and selection operator; MDS-UPDRS-III, Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale Part III; OH, orthostatic hypotension; PIGD, postural instability and gait difficulty; POSINS, initial postural instability symptom; QUIP, Questionnaire for Impulsive-Compulsive Disorders in Parkinson's Disease; RBDSQ, rapid eye movement sleep behavior disorder screening questionnaire; SCOPA-AUT, Scales for Outcomes in Parkinson's Disease-Autonomic; TD, tremor dominant. GBA and LRRK2 denote subgroup labels.",
    "tab:S12",
    [
        "Predictor",
        "Selected (of 500)",
        "Frequency (%)".replace("%", BS + "%"),
        "Final model",
    ],
    rows,
    "lccc",
    longtable=True,
)

# Export the saved reduced-model results without refitting or bootstrap.
if not FEATURE_TABLES_ONLY:
    from export_reduced_model_table import export_reduced_model_table

    export_reduced_model_table()

# ---------------------------------------------------------------- figures
if "--tables-only" in sys.argv or FEATURE_TABLES_ONLY:
    print("Supplementary tables written; figure and LASSO computation skipped.")
    sys.exit(0)

import matplotlib.pyplot as plt
import benchmark_evaluation as be
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import step_12_lasso as s12
import step_19_manuscript_figures as s19

preds7 = pd.read_csv("step_21_benchmark_predictions_7feat.csv")
y = preds7["MCI"].to_numpy().astype(int)
plot_dict = {
    m: preds7[f"prob_{m}"].to_numpy() for m in MODELS if f"prob_{m}" in preds7.columns
}
import figstyle as fs

fs.apply()


def diagnostics_figure():
    """Figure S2: ROC, precision-recall, calibration and decision curves of all models in a 2 x 2 grid."""
    from sklearn.calibration import calibration_curve

    fig, axes = fs.axes_grid(2, 2, hgap=17.0)
    (ax_roc, ax_pr), (ax_cal, ax_dca) = axes
    prev, n = y.mean(), len(y)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=0.9)
    ax_pr.axhline(prev, color="k", linestyle="--", alpha=0.5, linewidth=0.9)
    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=0.9)
    th = np.linspace(0.02, 0.70, 70)
    ax_dca.plot(th, np.zeros_like(th), "k--", linewidth=0.9, label="Treat none")
    ax_dca.plot(
        th,
        prev - (1 - prev) * th / (1 - th),
        color="grey",
        linestyle=":",
        linewidth=1.2,
        label="Treat all",
    )
    h_roc, h_pr, h_cal, names, aurocs, auprcs, slopes, briers = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for m, p in plot_dict.items():
        kw = dict(
            color=be.MODEL_COLORS.get(m, "#333333"),
            linestyle=be.MODEL_LINESTYLES.get(m, "-"),
            linewidth=1.2,
        )
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        h_roc.append(ax_roc.plot(fpr, tpr, **kw)[0])
        h_pr.append(ax_pr.plot(rec, prec, **kw)[0])
        true, pred = calibration_curve(y, p, n_bins=6, strategy="uniform")
        h_cal.append(ax_cal.plot(pred, true, marker="o", markersize=3, **kw)[0])
        nb = [
            ((p >= t) & (y == 1)).sum() / n
            - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)
            for t in th
        ]
        ax_dca.plot(th, nb, label=name(m), **kw)
        names.append(name(m))
        aurocs.append(f"{roc_auc_score(y, p):.4f}")
        auprcs.append(f"{average_precision_score(y, p):.4f}")
        slopes.append(f"{be.compute_calibration_slope_intercept(y, p)[0]:.2f}")
        briers.append(f"{brier_score_loss(y, p):.4f}")
    ax_roc.set_xlabel("1 $-$ specificity")
    ax_roc.set_ylabel("Sensitivity")
    ax_pr.set_xlabel("Recall (sensitivity)")
    ax_pr.set_ylabel("Precision (positive predictive value)")
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed proportion")
    ax_dca.set_xlabel("Threshold probability")
    ax_dca.set_ylabel("Net benefit")
    for ax in (ax_roc, ax_pr):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_dca.set_xlim(0.02, 0.70)
    ax_dca.set_ylim(-0.05, prev + 0.05)
    for ax, lab in ((ax_roc, "A"), (ax_pr, "B"), (ax_cal, "C"), (ax_dca, "D")):
        ax.grid(alpha=0.25)
        fs.panel_letter(ax, lab)
    # Legends with many rows: Arial table legends in a smaller size than in the other figures
    fs.table_legend(
        ax_roc,
        h_roc,
        names,
        [("AUROC", aurocs)],
        "Model",
        fontsize=4.6,
        loc="lower right",
    )
    fs.table_legend(
        ax_pr,
        h_pr,
        names,
        [("AUPRC", auprcs)],
        "Model",
        fontsize=4.6,
        loc="upper right",
    )
    fs.table_legend(
        ax_cal,
        h_cal,
        names,
        [("Slope", slopes), ("Brier", briers)],
        "Model",
        fontsize=4.4,
        loc="upper left",
    )
    ax_dca.legend(loc="upper right", ncol=2, frameon=True, fontsize=5.0)
    fig.savefig(
        os.path.join(FIG, "figS_diagnostics_7feat.pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


@s12.with_cm_math
def lasso_figure():
    """Redraw saved LASSO paths and the existing A-panel display curve, without CV."""
    curve = pd.read_csv("lasso_display_curve.csv")
    provenance = json.load(
        open("lasso_display_curve_provenance.json", encoding="utf-8")
    )
    path = pd.read_csv("step_12_lasso_coefficients_path.csv", index_col=0)
    cols = [
        c
        for c in pd.read_csv("PPMI_8_train.csv", nrows=0).columns
        if c not in ("SITE", "subgroup", "EVENT_ID", "visit_date", "PATNO", "MCI")
    ]
    if path.shape[1] != len(cols) or len(cols) != 28:
        raise ValueError("Expected the saved coefficient paths for all 28 predictors")
    maximum = 10**2.2
    show = curve["lambda"].to_numpy() <= maximum * (1 + 1e-9)
    lam = curve["lambda"].to_numpy()
    mean = curve["mean_aucpr_display"].to_numpy()
    ib, i1 = provenance["best_index"], provenance["one_se_index"]
    fig, axes = fs.axes_grid(1, 2, ax_w=64.0, ax_h=46.0, right=3.0, left=14.0)
    ax1, ax2 = axes[0]
    ax1.plot(
        lam[show], mean[show], "-o", linewidth=1.2, markersize=2.2, color="#1f77b4"
    )
    ax1.fill_between(
        lam[show],
        curve["lower_display"].to_numpy()[show],
        curve["upper_display"].to_numpy()[show],
        alpha=0.2,
        color="#1f77b4",
    )
    handles = []
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlim(1, maximum)
        ax.set_xlabel(r"$\lambda$")
        ax.grid(alpha=0.3)
        best = ax.axvline(lam[ib], color="r", linestyle="--", alpha=0.6, linewidth=0.9)
        one_se = ax.axvline(
            lam[i1], color="g", linestyle="--", alpha=0.6, linewidth=0.9
        )
        if ax is ax1:
            handles = [best, one_se]
    ax1.plot(lam[ib], mean[ib], "ro", markersize=4)
    ax1.plot(lam[i1], mean[i1], "go", markersize=4)
    ax1.set_ylabel("AUPRC")
    fs.table_legend(
        ax1,
        handles,
        ["Best", "1SE"],
        [("AUPRC", [f"{mean[ib]:.4f}", f"{mean[i1]:.4f}"])],
        header="Rule",
        fontsize=fs.LEGEND_SIZE,
        loc="lower left",
    )
    coefficient_lambda = path.index.to_numpy(dtype=float)
    keep = coefficient_lambda <= maximum * (1 + 1e-9)
    for i in range(len(cols)):
        ax2.plot(
            coefficient_lambda[keep], path.iloc[keep, i], linewidth=1.1, alpha=0.85
        )
    ax2.set_ylabel("Coefficient")
    fs.panel_letter(ax1, "A")
    fs.panel_letter(ax2, "B")
    fig.savefig(
        os.path.join(FIG, "figS_lasso.pdf"), bbox_inches="tight", pad_inches=0.03
    )
    plt.close(fig)


diagnostics_figure()
if "--skip-lasso" not in sys.argv:
    lasso_figure()
s19.delta_forest(
    "step_15_benchmark_bootstrap_diffs_full.csv",
    os.path.join(FIG, "figS_forest_28feat.pdf"),
    auprc_title="ΔAUPRC vs LR",
)
shutil.copyfile(
    "step_17_shap_beeswarm_all_7feat.pdf",
    os.path.join(FIG, "figS_shap_beeswarm_all.pdf"),
)
print("Wrote supplementary tables and figures to", os.path.abspath(OUT))
