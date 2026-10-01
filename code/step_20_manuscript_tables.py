"""Step 20: Manuscript tables 1-3 as LaTeX, typeset from saved results (nothing is re-estimated).

Outputs (in ../manuscript/tables/): table1.tex, table2.tex, table3.tex
"""

from __future__ import annotations

import os
import re

import pandas as pd

OUT_DIR = os.path.join("..", "manuscript", "tables")
os.makedirs(OUT_DIR, exist_ok=True)


def write(name: str, text: str) -> None:
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


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
    ("Age, years", "Age"), ("Male sex", "Sex|Male"), ("Education, years", "Education"),
    ("Disease duration, years", "Duration"), ("MDS-UPDRS III", "UPDRS-III"),
    ("GDS score", "GDS"), ("UPSIT percentile", "upsit_pctl"),
    ("Caudate SBR, mean of both sides", "MIA_CAUDATE_mean"),
    ("Family history of PD", "FH-PD|Yes"), ("Other initial symptom", "Other symptoms|Yes"),
]
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Baseline characteristics of the training and test sets.} Continuous variables are median (interquartile range); categorical variables are n (\%).}",
         r"\label{tab:baseline}", r"\begin{tabular}{lcc}", r"\toprule",
         r" & Training set & Test set \\", r" & (n = 1,165) & (n = 500) \\", r"\midrule"]
for lab, key in rows1:
    lines.append(f"{lab} & {esc(str(tr[key]))} & {esc(str(te[key]))} \\\\")
lines.append(r"PD-MCI & 183 (15.7\%) & 78 (15.6\%) \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("table1.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table 2
def f3(x):
    t = f"{x:.5f}" if 0 < abs(x) < 0.0005 else f"{x:.3f}"
    return t.replace("-", "$-$")


orr = pd.read_csv("step_13_lr_or_table.csv", encoding="utf-8-sig").set_index("Variable")
shr = pd.read_csv("step_14_shrunk_coefficients.csv").set_index("Variable")
shr_key = {"upsit_pctl": "upsit_pctl", "MIA_CAUDATE_mean": "MIA_CAUDATE_mean"}
spec = [
    ("Male sex", "SEX", "SEX", "vs female"),
    ("Education", "EDUCYRS", "EDUCYRS", "per year"),
    ("GDS score", "gds", "gds", "per point"),
    ("Caudate SBR (mean)", "MIA_CAUDATE_mean", "MIA_CAUDATE_mean", "per 1.0 unit"),
    ("UPSIT percentile", "upsit_pctl", "upsit_pctl", "per percentile"),
    ("Other initial symptom", "sym_other", "sym_other", "yes vs no"),
    ("Family history of PD", "fampd_bin", "fampd_bin", "yes vs no"),
]
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Final logistic regression model for concurrent PD-MCI.} The final model uses the shrunk coefficients (uniform shrinkage factor 0.814). Odds ratios (ORs) refer to the unit shown. Unshrunk ORs with 95\% CIs and $P$ values are from the unpenalized fit before shrinkage.}",
         r"\label{tab:model}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
         r"Predictor & Unit & Unshrunk OR (95\% CI) & $P$ & Shrunk $\beta$ & Shrunk OR \\", r"\midrule"]
for lab, k13, k14, unit in spec:
    r13 = orr.loc[k13]
    m_ = re.match(r"([\d.]+) \(([\d.]+)–([\d.]+)\)", str(r13["OR (95% CI)"]))
    ors = f"{float(m_[1]):.2f} ({float(m_[2]):.2f}, {float(m_[3]):.2f})"
    p = "$<$0.001" if str(r13["p"]).startswith("<") else f"{float(r13['p']):.3f}"
    r14 = shr.loc[[i for i in shr.index if i == k14 or shr.loc[i, "Abbreviation"] == k14][0]]
    if k13 == "upsit_pctl":
        ors = f"{float(m_[1]):.3f} ({float(m_[2]):.3f}, {float(m_[3]):.3f})"
    lines.append(f"{lab} & {unit} & {ors} & {p} & {f3(r14['beta_shrunk'])} & {r14['OR_shrunk']:.3f} \\\\")
b0 = shr.loc["Intercept", "beta_shrunk"]
lines.append(f"Intercept & & & & {f3(b0)} & \\\\")
lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
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
    lo, hi = float(s[:i]), float(s[i + 1:])
    return f"{f3(d)} ({f3(lo)}, {f3(hi)})"


lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
         r"\caption{\textbf{Test-set performance of the final logistic regression (LR) model and ten comparator models using the same seven predictors.} The test set comprised $n = 500$ participants, 78 of them with PD-MCI; the AUPRC of a non-informative model is 0.156. $\Delta$ is the model minus LR on the same participants (paired bootstrap, 2,000 resamples); 95\% CIs are percentile intervals and $P$ is for $\Delta$AUPRC. Calibration slope is from logistic recalibration of the raw predicted probabilities. No adjustment was made for multiple comparisons.}",
         r"\label{tab:performance}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
         r"Model & AUPRC (95\% CI) & $\Delta$AUPRC (95\% CI) & $P$ & AUROC (95\% CI) & $\Delta$AUROC (95\% CI) & Brier & Calibration slope \\",
         r"\midrule"]
for m in order:
    r = ci.loc[m]
    a = f"{f3(r.AUPRC)} ({f3(r.AUPRC_CI_lower)}, {f3(r.AUPRC_CI_upper)})"
    u = f"{f3(r.AUROC)} ({f3(r.AUROC_CI_lower)}, {f3(r.AUROC_CI_upper)})"
    if m == "LR":
        da, p, du = "--", "--", "--"
    else:
        d = df.loc[m]
        da, du = dfmt(d.Delta_AUPRC, d.Delta_AUPRC_95CI), dfmt(d.Delta_AUROC, d.Delta_AUROC_95CI)
        p = f"{d.Delta_AUPRC_pval:.3f}"
    lines.append(f"{lab.get(m, m)} & {a} & {da} & {p} & {u} & {du} & {r.Brier:.3f} & {r.Calib_Slope:.2f} \\\\")
    if m == "LR":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
write("table3.tex", "\n".join(lines) + "\n")
print("Wrote tables 1-3 to", os.path.abspath(OUT_DIR))
