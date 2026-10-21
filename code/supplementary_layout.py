"""Supplementary table layout; file IDs predate printed table renumbering."""

import re


def _compact_longtable(text, spec, separation):
    start = text.index(r"\multicolumn{")
    end = text.index(r"\endlastfoot", start)
    footer = text[start:end]
    if r"\ManuscriptTableNotes{" in footer:
        opening = footer.index(r"\ManuscriptTableNotes{") + len(
            r"\ManuscriptTableNotes"
        )
        depth, closing = 1, opening + 1
        while depth:
            depth += (footer[closing] == "{") - (footer[closing] == "}")
            closing += 1
        note = footer[opening + 1 : closing - 1]
    else:
        note = footer.split(r"\textbf{Note:}", 1)[1].split(r"\end{minipage}", 1)[0]
        note = r"\textbf{Note:}" + note
    text = text[:start] + text[end:]
    text = re.sub(
        r"\\begin\{longtable\}\{[^\n]+",
        lambda m: r"\setlength{\tabcolsep}{"
        + separation
        + "}\n"
        + r"\begin{longtable}{"
        + spec
        + "}",
        text,
    )
    return text.replace(
        r"\end{longtable}",
        r"\end{longtable}" + "\n" + r"\ManuscriptTableNotes{" + note + "}",
        1,
    )


def supplementary_layout(fname, text):
    """Change presentation only, after the standard caption/notes conversion."""
    if fname in {"S9.tex", "S10.tex"}:  # printed Tables S8/S9
        text = re.sub(r"(?<=\d)--(?=\d)", " to ", text)
    if fname == "S7.tex":  # printed Table S6: three-decimal CIs
        text = text.replace(
            r"\begin{tabular*}",
            r"\setlength{\tabcolsep}{3.5pt}" + "\n" + r"\begin{tabular*}",
            1,
        )
    if fname == "S3.tex":  # printed Table S2
        text = re.sub(
            r"\\begin\{longtable\}\{[^\n]+",
            lambda m: r"\begin{longtable}{@{\extracolsep{\fill}}l>{\raggedright\arraybackslash}p{6.7cm}cccc}",
            text,
        )
        for old, new in [
            (r"\makecell{Missing,\\ training (\%)}", r"\makecell{Train\\n (\%)}"),
            (r"\makecell{Missing,\\ test (\%)}", r"\makecell{Test\\n (\%)}"),
            (r"\makecell{Value used for\\ missing data}", "Imputation"),
            ("Selected", "LASSO"),
        ]:
            text = text.replace(old, new)
        text = re.sub(r"\\ManuscriptTableSize\{[^}]+\}\{[^}]+\}\n", "", text)
        text = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]+\}\n", "", text)
        text = text.replace(
            r"\begin{longtable}",
            r"\ManuscriptTableSize{8.5}{10.5}"
            + "\n"
            + r"\setlength{\tabcolsep}{2pt}"
            + "\n"
            + r"\begin{longtable}",
            1,
        )
    elif fname == "S4.tex":  # printed Table S3
        text = re.sub(
            r"\\begin\{longtable\}\{[^\n]+",
            lambda m: r"\begin{longtable}{@{\extracolsep{\fill}}ll>{\raggedright\arraybackslash}p{9.2cm}}",
            text,
        )
    elif fname in {"S5.tex", "S12.tex"}:  # printed Tables S4 and S11
        spec = "@{}lllll@{}" if fname == "S5.tex" else "@{}lccc@{}"
        separation = "4pt" if fname == "S5.tex" else "6pt"
        text = re.sub(r"\\setlength\{\\LTleft\}[^\n]*\n", "", text)
        text = text.replace(
            r"\begin{longtable}",
            r"\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}"
            + "\n"
            + r"\begin{longtable}",
            1,
        )
        text = _compact_longtable(text, spec, separation)
    elif fname == "S8.tex":  # printed Table S7: models in rows
        text = re.sub(r"\\resizebox\{\\textwidth\}\{!\}\{%\n", "", text)
        text = text.replace(r"\end{tabular}}", r"\end{tabular}")
        mid = text.index(r"\midrule") + len(r"\midrule")
        end = text.index(r"\bottomrule", mid)
        rows = [
            line.strip()[:-2].strip().split(" & ")
            for line in text[mid:end].strip().splitlines()
            if " & " in line
        ]
        assert len(rows) == 7 and all(len(row) == 12 for row in rows)
        # Model labels have discretionary line breaks only in the old header.
        models = [
            "LR",
            "SVM",
            "RF",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "RealMLP",
            "TabM",
            "TabICL v2.0",
            "TabPFN v3.5",
            "TabFM v1.0",
        ]
        header = (
            " & ".join(
                r"\ManuscriptHeader{" + x + "}"
                for x in ["Model"] + [row[0] for row in rows]
            )
            + r" \\"
        )
        body = "\n".join(
            " & ".join([model] + [row[i + 1] for row in rows]) + r" \\"
            for i, model in enumerate(models)
        )
        top = text.index(r"\toprule") + len(r"\toprule")
        text = (
            text[:top]
            + "\n"
            + header
            + "\n"
            + r"\midrule"
            + "\n"
            + body
            + "\n"
            + text[end:]
        )
        text = re.sub(
            r"\\begin\{tabular\}\{[^\n]+",
            lambda m: r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc}",
            text,
        )
        text = text.replace(r"\end{tabular}", r"\end{tabular*}")
        if r"\ManuscriptTableSize" not in text:
            text = text.replace(
                r"\begin{tabular*}",
                r"\ManuscriptTableSize{8}{10}"
                + "\n"
                + r"\setlength{\tabcolsep}{1.5pt}"
                + "\n"
                + r"\begin{tabular*}",
                1,
            )
        text = text.replace(
            "Predictors are ordered by their rank in LR.",
            "Predictor columns are ordered by their rank in LR; models are listed in rows.",
        )
    elif fname == "S9.tex":  # printed Table S8
        text = re.sub(r"\\resizebox\{\\textwidth\}\{!\}\{%\n", "", text)
        text = text.replace(r"\end{tabular}}", r"\end{tabular}")
        text = re.sub(
            r"\\begin\{tabular\}\{[^\n]+",
            lambda m: r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc}",
            text,
        )
        text = text.replace(r"\end{tabular}", r"\end{tabular*}")
        text = text.replace(
            r"\ManuscriptTableSize{8.5}{10.5}", r"\ManuscriptTableSize{7.5}{9.5}"
        )
        text = text.replace(
            r"\ManuscriptTableSize{8}{10}", r"\ManuscriptTableSize{7.5}{9.5}"
        )
        if r"\ManuscriptTableSize" not in text:
            text = text.replace(r"\toprule", r"\toprule", 1).replace(
                r"\begin{tabular*}",
                r"\ManuscriptTableSize{7.5}{9.5}"
                + "\n"
                + r"\setlength{\tabcolsep}{2pt}"
                + "\n"
                + r"\begin{tabular*}",
                1,
            )
        text = re.sub(
            r"\\setlength\{\\tabcolsep\}\{[^}]+\}",
            lambda m: r"\setlength{\tabcolsep}{1.4pt}",
            text,
        )
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if " & " in line and not ("Header" in line or line.startswith("Model &")):
                cells = line[:-2].strip().split(" & ")
                if len(cells) == 9:
                    for j in (1, 2, 4, 5, 7):
                        if " (" in cells[j]:
                            estimate, interval = cells[j].split(" (", 1)
                            cells[j] = (
                                r"\makecell{" + estimate + r"\\(" + interval + "}"
                            )
                    lines[i] = " & ".join(cells) + r" \\"
        text = "\n".join(lines) + "\n"
    elif fname == "S11.tex":  # printed Table S10
        text = text.replace(
            "Refit (probability)", r"\makecell{Refit\\(probability)}"
        ).replace("Additivity (log-odds)", r"\makecell{Additivity\\(log-odds)}")
    return text
