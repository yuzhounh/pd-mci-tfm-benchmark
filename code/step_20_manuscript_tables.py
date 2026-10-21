"""Step 20: Manuscript tables 1-3 as LaTeX, typeset from saved results (nothing is re-estimated).

Outputs (in ../manuscript/tables/): table1.tex, table2.tex, table3.tex
"""

from __future__ import annotations

import os
import re

import pandas as pd
from result_formatting import ci_pair, difference
from feature_labels import feature_label, PREDICTOR_NOTE, PREDICTOR_ABBREVIATIONS
from table_presentation import format_table_notes

OUT_DIR = os.path.join("..", "manuscript", "tables")
os.makedirs(OUT_DIR, exist_ok=True)


def write(name: str, text: str) -> None:
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(format_table_notes(text))


def esc(s: str) -> str:
    return s.replace("%", r"\%").replace("&", r"\&")


# ---------------------------------------------------------------- Table 1
def load_stats(split: str) -> dict[str, str]:
    d = pd.read_csv(f"step_09_characteristics_stats_{split}.csv", encoding="utf-8-sig")
    out, parent = {}, None
    for _, r in d.iterrows():
        v = str(r["Variable"])
        if v.startswith("  "):
            out[f"{parent}|{v.strip()}"] = r["Overall"]
        else:
            parent = v
            out[v] = r["Overall"]
    return out


tr, te = load_stats("train"), load_stats("test")
rows1 = [
    ("Age", "Age"),
    (feature_label("SEX"), "Sex|Male"),
    (feature_label("EDUCYRS"), "Education"),
    ("Duration", "Duration"),
    (feature_label("updrs3_score"), "UPDRS-III"),
    (feature_label("gds"), "GDS"),
    (feature_label("upsit_pctl"), "upsit_pctl"),
    (feature_label("MIA_CAUDATE_mean"), "MIA_CAUDATE_mean"),
    (feature_label("fampd_bin"), "FH-PD|Yes"),
    (feature_label("sym_other"), "Other symptoms|Yes"),
]
lines = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\small",
    r"\caption{\textbf{Baseline characteristics.}}",
    r"\label{tab:baseline}",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r" & Training set & Test set \\",
    r" & (n = 1,165) & (n = 500) \\",
    r"\midrule",
]
for lab, key in rows1:
    lines.append(
        f"{lab} & {esc(str(tr[key]).replace('%', ''))} & {esc(str(te[key]).replace('%', ''))} \\\\"
    )
lines.append(r"PD-MCI & 183 (15.7) & 78 (15.6) \\")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\medskip",
    r"\begin{minipage}{\textwidth}",
    r"\footnotesize\raggedright",
    r"\textit{Note:} Training and test sets are summarized after training-derived imputation; pre-imputation missingness is reported in Supplementary Table~S2. Continuous variables are median (interquartile range); categorical variables are n (\%). Sex reports males; FH-PD and Other symptom report the yes category; PD-MCI reports participants with PD-MCI. Age, Education and Duration are in years; Education is capped at 20. GDS is the 15-item score, UPSIT percentile is age- and sex-adjusted, and Caudate SBR is the bilateral mean. Other symptom denotes the other initial symptom at diagnosis flag. \textit{Abbreviations:} "
    + PREDICTOR_ABBREVIATIONS
    + r" MDS-UPDRS-III, Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale Part III; PD-MCI, mild cognitive impairment in Parkinson's disease.",
    r"\end{minipage}",
    r"\end{table}",
]
write("table1.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- Table 2
def f3(x):
    t = f"{x:.5f}" if 0 < abs(x) < 0.0005 else f"{x:.3f}"
    return t.replace("-", "$-$")


orr = pd.read_csv("step_13_lr_or_table.csv", encoding="utf-8-sig").set_index("Variable")
shr = pd.read_csv("step_14_shrunk_coefficients.csv").set_index("Variable")
shr_key = {"upsit_pctl": "upsit_pctl", "MIA_CAUDATE_mean": "MIA_CAUDATE_mean"}
spec = [
    (feature_label("SEX"), "SEX", "SEX", "male vs female"),
    (feature_label("EDUCYRS"), "EDUCYRS", "EDUCYRS", "per year"),
    (feature_label("gds"), "gds", "gds", "per point"),
    (
        feature_label("MIA_CAUDATE_mean"),
        "MIA_CAUDATE_mean",
        "MIA_CAUDATE_mean",
        "per 1.0 unit",
    ),
    (feature_label("upsit_pctl"), "upsit_pctl", "upsit_pctl", "per percentile point"),
    (feature_label("sym_other"), "sym_other", "sym_other", "yes vs no"),
    (feature_label("fampd_bin"), "fampd_bin", "fampd_bin", "yes vs no"),
]
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    "\\caption{\\textbf{Final logistic regression model for concurrent PD-MCI.} The final model uses shrunk coefficients (shrinkage factor 0.814) and original-scale predictors, without z-standardization; Equation~\\eqref{eq:finalmodel} gives probabilities and input coding. Continuous ORs are per year of Education (capped at 20), GDS point (15-item score), one unit of bilateral Caudate SBR, or UPSIT percentile point (adjusted for age and sex). Binary ORs compare male versus female for Sex and yes versus no for FH-PD and Other symptom. Unshrunk ORs, 95\\% CIs and $P$ values describe the selected unpenalized model before shrinkage, exclude selection uncertainty and are not confirmatory association tests. Other symptom denotes the other initial symptom at diagnosis flag. \\textbf{Abbreviations:} "
    + PREDICTOR_ABBREVIATIONS
    + " CI, confidence interval; OR, odds ratio; PD-MCI, mild cognitive impairment in Parkinson's disease.}",
    r"\label{tab:model}",
    r"\ManuscriptTableSize{10}{12}",
    r"\ManuscriptTableStyle",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    "\\ManuscriptHeader{Predictor} & \\ManuscriptHeader{Unshrunk OR (95\\% CI)} & \\ManuscriptHeader{$P$} & \\ManuscriptHeader{Shrunk $\\beta$} & \\ManuscriptHeader{Shrunk OR} \\\\",
    r"\midrule",
]
for lab, k13, k14, unit in spec:
    r13 = orr.loc[k13]
    m_ = re.match(r"([\d.]+) \(([\d.]+)–([\d.]+)\)", str(r13["OR (95% CI)"]))
    ors = f"{float(m_[1]):.2f} " + ci_pair(f"{float(m_[2]):.2f}", f"{float(m_[3]):.2f}")
    p = "$<$0.001" if str(r13["p"]).startswith("<") else f"{float(r13['p']):.3f}"
    r14 = shr.loc[
        [i for i in shr.index if i == k14 or shr.loc[i, "Abbreviation"] == k14][0]
    ]
    if k13 == "upsit_pctl":
        ors = f"{float(m_[1]):.3f} " + ci_pair(
            f"{float(m_[2]):.3f}", f"{float(m_[3]):.3f}"
        )
    lines.append(
        f"{lab} & {ors} & {p} & {f3(r14['beta_shrunk'])} & {r14['OR_shrunk']:.3f} \\\\"
    )
b0 = shr.loc["Intercept", "beta_shrunk"]
lines.append(f"Intercept & & & {f3(b0)} & \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("table2.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table 3
ci = pd.read_csv("step_21_benchmark_model_ci_7feat.csv").set_index("Model")
df = pd.read_csv("step_21_benchmark_bootstrap_diffs_7feat.csv").set_index("Model")
order = ["LR"] + list(ci.drop("LR").sort_values("AUPRC", ascending=False).index)
lab = {"TabICLv2": "TabICL v2.0", "TabPFN": "TabPFN v3.5", "TabFM": "TabFM v1.0"}


def cistr(s):
    lo, hi = str(s).replace("−", "-").split("–") if "–" in str(s)[1:] else (None, None)
    return lo, hi


def dfmt(d, ci_s):
    s = str(ci_s)
    i = s.index("–", 1)
    lo, hi = float(s[:i]), float(s[i + 1 :])
    return f"{f3(d)} " + ci_pair(f3(lo), f3(hi))


lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\footnotesize",
    r"\caption{\textbf{Test-set performance of the final logistic regression (LR) model and ten comparator models using the same seven predictors.} The test set comprised $n = 500$ participants, 78 of them with PD-MCI; the AUPRC of a non-informative model is 0.156 and the Brier score of predicting the prevalence for every participant is 0.132. 95\% CIs are percentile intervals. $P$ compares each model's AUPRC with that of LR on the same participants (paired bootstrap, 2,000 resamples). Calibration slope is from logistic recalibration of the raw predicted probabilities. No adjustment was made for multiple comparisons.}",
    r"\label{tab:performance}",
    r"% Keep an explicit table font instead of scaling the entire table.",
    r"\begin{tabular}{lccccc}",
    r"\toprule",
    r"Model & AUPRC (95\% CI) & $P$ & AUROC (95\% CI) & Brier & Calibration slope \\",
    r"\midrule",
]
for m in order:
    r = ci.loc[m]
    a = f"{f3(r.AUPRC)} " + ci_pair(f3(r.AUPRC_CI_lower), f3(r.AUPRC_CI_upper))
    u = f"{f3(r.AUROC)} " + ci_pair(f3(r.AUROC_CI_lower), f3(r.AUROC_CI_upper))
    if m == "LR":
        p = "--"
    else:
        d = df.loc[m]
        p = f"{d.Delta_AUPRC_pval:.3f}"
    lines.append(
        f"{lab.get(m, m)} & {a} & {p} & {u} & {r.Brier:.3f} & {r.Calib_Slope:.2f} \\\\"
    )
    if m == "LR":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("table3.tex", "\n".join(lines) + "\n")
print("Wrote tables 1-3 to", os.path.abspath(OUT_DIR))
