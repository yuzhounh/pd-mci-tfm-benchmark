"""Export Supplementary Table S12 from saved reduced-LR results; no computation."""

from pathlib import Path
import pandas as pd
from result_formatting import number

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("7 predictors (final)", "Final LR"),
    ("without Other symptom", "Reduced LR 1"),
    ("without Other symptom and FH-PD", "Reduced LR 2"),
]


def fixed(value, places=3):
    return f"{value:.{places}f}".replace("-", r"$-$")


def estimate_ci(row, metric):
    return (
        fixed(row[metric])
        + " ("
        + fixed(row[metric + "_ci_low"])
        + " to "
        + fixed(row[metric + "_ci_high"])
        + ")"
    )


def brier_difference(row):
    return (
        number(row["Delta_Brier"])
        + " ("
        + number(row["Delta_Brier_ci_low"])
        + " to "
        + number(row["Delta_Brier_ci_high"])
        + ")"
    )


def split_estimate_interval(value):
    if " (" not in value:
        return value
    point, interval = value.split(" (", 1)
    return r"\makecell{" + point + r"\\(" + interval + "}"


def export_reduced_model_table(destination=None):
    data = pd.read_csv(ROOT / "code/step_23_lr_reduced_sensitivity.csv").set_index(
        "model"
    )
    header = [
        r"\ManuscriptHeader{Model}",
        r"\ManuscriptHeader{AUPRC}",
        r"\ManuscriptHeader{\makecell{$\Delta$\\AUPRC}}",
        r"\ManuscriptHeader{AUROC}",
        r"\ManuscriptHeader{\makecell{$\Delta$\\AUROC}}",
        r"\ManuscriptHeader{Brier}",
        r"\ManuscriptHeader{\makecell{$\Delta$\\Brier}}",
        r"\ManuscriptHeader{Slope}",
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{\textbf{Exploratory sensitivity of the final LR to removing the less stable predictors.}}",
        r"\label{tab:reduced}",
        r"\ManuscriptTableSize{8}{10}",
        r"\setlength{\tabcolsep}{1.4pt}",
        r"\ManuscriptTableStyle",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for i, (key, label) in enumerate(MODELS):
        row = data.loc[key]
        cells = [
            label,
            estimate_ci(row, "AUPRC"),
            "--" if i == 0 else estimate_ci(row, "Delta_AUPRC"),
            estimate_ci(row, "AUROC"),
            "--" if i == 0 else estimate_ci(row, "Delta_AUROC"),
            fixed(row["Brier"]),
            "--" if i == 0 else brier_difference(row),
            fixed(row["Calibration_slope"], 2),
        ]
        for j in (1, 2, 3, 4, 6):
            cells[j] = split_estimate_interval(cells[j])
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\ManuscriptTableNotes{\textbf{Note:} Values are estimates with 95\% CIs below. "
        r"Final LR is the full seven-predictor model. Reduced LR 1 omits Other symptom (six predictors); "
        r"Reduced LR 2 omits Other symptom and FH-PD (five predictors). "
        r"This was a post hoc analysis on the test set ($n = 500$; 78 with PD-MCI). "
        r"Each reduced model was refitted on the training set with the same estimator and the locked "
        r"shrinkage factor of the final model (0.814; intercept re-estimated with the shrunk linear predictor as offset). "
        r"$\Delta$ is the reduced model minus Final LR, estimated by the same participant-level paired bootstrap "
        r"(2,000 resamples); positive $\Delta$Brier indicates worse performance. Slope: calibration slope.\par",
        r"\noindent\textbf{Abbreviations:} AUPRC, area under the precision--recall curve; "
        r"AUROC, area under the receiver operating characteristic curve; CI, confidence interval; "
        r"FH-PD, family history of Parkinson's disease; LR, logistic regression; "
        r"PD-MCI, mild cognitive impairment in Parkinson's disease.}",
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    output = (
        Path(destination)
        if destination
        else ROOT / "manuscript/tables/reduced_model_sensitivity.tex"
    )
    output.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    export_reduced_model_table()
