"""Manuscript display names; analysis column names and legacy result keys stay fixed."""

from __future__ import annotations

import csv
from pathlib import Path

with Path(__file__).with_suffix(".csv").open(encoding="utf-8", newline="") as source:
    FEATURE_DEFINITIONS = {row["feature"]: row for row in csv.DictReader(source)}
FEATURE_LABELS = {key: row["label"] for key, row in FEATURE_DEFINITIONS.items()}


def feature_label(key: str) -> str:
    return FEATURE_LABELS[key]


PREDICTOR_NOTE = (
    "Education is in years (capped at 20), GDS is the 15-item score, UPSIT is the age- and "
    "sex-adjusted percentile, and Caudate SBR is the bilateral mean. Other symptom denotes "
    "the other initial symptom at diagnosis flag. Sex: 0 = female, 1 = male; FH-PD and "
    "Other symptom: 0 = no, 1 = yes."
)
PREDICTOR_ABBREVIATIONS = (
    "FH-PD, family history of Parkinson's disease; GDS, Geriatric Depression Scale; "
    "SBR, specific binding ratio; UPSIT, University of Pennsylvania Smell Identification Test."
)
