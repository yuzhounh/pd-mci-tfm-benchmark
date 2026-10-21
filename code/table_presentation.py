"""Keep explanatory notes below tables when saved-result tables are exported."""

from __future__ import annotations

import re


def _braced(text: str, opening: int) -> tuple[str, int]:
    depth = 0
    for i in range(opening, len(text)):
        if i and text[i - 1] == "\\":
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if not depth:
                return text[opening + 1 : i], i + 1
    raise ValueError("Unclosed TeX argument")


def _note_labels(text: str) -> str:
    text = re.sub(r"\\text(?:it|bf)\{\s*Note:\}", r"\\textbf{Note:}", text)
    return re.sub(
        r"\s*\\text(?:it|bf)\{\s*Abbreviations:\}",
        r"\\par\n\\textbf{Abbreviations:}",
        text,
    )


def format_table_notes(text: str) -> str:
    """Preserve titles, labels, table rows and note wording; change placement/style."""
    match = re.search(r"\\caption\{", text)
    if match is None:
        return text
    caption, end = _braced(text, match.end() - 1)
    if not caption.startswith(r"\textbf{"):
        raise ValueError("Expected a bold short table title")
    title, title_end = _braced(caption, len(r"\textbf"))
    note = caption[title_end:].strip()
    labels = re.findall(r"\\label\{[^}]+\}", note)
    note = re.sub(r"\\label\{[^}]+\}", "", note).strip()
    replacement = r"\caption{\textbf{" + title + "}" + "".join(labels) + "}"
    text = text[: match.start()] + replacement + text[end:]
    # Table 1 already has a note block below its body.
    if not note:
        existing = re.search(
            r"\\par\\medskip\s*\\begin\{minipage\}\{\\textwidth\}\s*"
            r"\\footnotesize\\raggedright\s*(.*?)\\end\{minipage\}",
            text,
            re.S,
        )
        if existing:
            block = (
                r"\ManuscriptTableNotes{"
                + _note_labels(existing.group(1).strip())
                + "}\n"
            )
            text = text[: existing.start()] + block + text[existing.end() :]
        return text
    note = _note_labels(r"\textbf{Note:} " + note)
    block = r"\ManuscriptTableNotes{" + note + "}"
    if r"\begin{longtable}" in text:
        # A last-page footer keeps the note attached to the last data rows.
        header = text.split(r"\toprule", 1)[1].splitlines()[1]
        columns = header.count(" & ") + 1
        footer = (
            r"\endfoot"
            + "\n"
            + r"\bottomrule"
            + "\n"
            + r"\multicolumn{"
            + str(columns)
            + r"}{@{}p{\textwidth}@{}}{"
            + block
            + r"} \\"
            + "\n"
            + r"\endlastfoot"
        )
        text = text.replace(r"\endfoot", footer, 1)
    else:
        text = text.replace(r"\end{table}", block + "\n" + r"\end{table}", 1)
    return text
