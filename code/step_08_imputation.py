"""Training-derived imputation for the baseline single-visit analysis.

No value is borrowed from another visit. Imputation parameters are estimated
from the training set only and then applied unchanged to the held-out test set.
"""

import json

import numpy as np
import pandas as pd


TRAIN_INPUT = "PPMI_6_train.csv"
TEST_INPUT = "PPMI_6_test.csv"
TRAIN_OUTPUT = "PPMI_7_train.csv"
TEST_OUTPUT = "PPMI_7_test.csv"
SUMMARY_OUTPUT = "step_08_imputation_summary.csv"
PARAMETERS_OUTPUT = "step_08_imputation_parameters.json"

IDENTIFIER_COLUMNS = ["PATNO", "MCI"]
CATEGORICAL_METADATA = ["SITE", "subgroup", "EVENT_ID", "visit_date"]
BINARY_FEATURES = {
    "SEX",
    "fampd_bin",
    "sym_tremor",
    "sym_rigid",
    "sym_brady",
    "sym_posins",
    "sym_other",
    "PDTRTMNT",
    "quip",
    "ess",
    "rem",
    "orthostasis",
    "td_pigd",
    "APOE_e4",
    "has_GBA",
    "has_LRRK2",
}


def validate_single_visit(df: pd.DataFrame, name: str) -> None:
    required = {"PATNO", "EVENT_ID", "MCI"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}")
    if not df["EVENT_ID"].eq("BL").all():
        raise RuntimeError(f"{name} contains non-baseline EVENT_ID values")
    if df["PATNO"].duplicated().any():
        raise RuntimeError(f"{name} contains duplicate PATNO values")
    if df[IDENTIFIER_COLUMNS].isna().any().any():
        raise RuntimeError(f"{name} contains missing PATNO or MCI values")


def training_mode(series: pd.Series):
    values = series.dropna().mode()
    if values.empty:
        raise RuntimeError(f"Cannot estimate a mode for all-missing column {series.name}")
    return values.iloc[0]


def training_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    if pd.isna(median):
        raise RuntimeError(f"Cannot estimate a median for all-missing column {series.name}")
    return float(median)


def main() -> None:
    print("=== Step 08: Baseline Single-Visit Imputation ===")
    train = pd.read_csv(TRAIN_INPUT)
    test = pd.read_csv(TEST_INPUT)
    validate_single_visit(train, "Training set")
    validate_single_visit(test, "Test set")

    if list(train.columns) != list(test.columns):
        raise RuntimeError("Training and test columns differ")

    before_train = train.isna().sum()
    before_test = test.isna().sum()
    parameters = {}

    for column in train.columns:
        if column in IDENTIFIER_COLUMNS:
            continue

        if column in CATEGORICAL_METADATA or column in BINARY_FEATURES:
            value = training_mode(train[column])
            strategy = "training_mode"
        else:
            value = training_median(train[column])
            strategy = "training_median"

        train[column] = train[column].fillna(value)
        test[column] = test[column].fillna(value)
        parameters[column] = {
            "strategy": strategy,
            "value": value.item() if isinstance(value, np.generic) else value,
        }

    remaining_train = train.isna().sum()
    remaining_test = test.isna().sum()
    if remaining_train.any() or remaining_test.any():
        unresolved = sorted(
            set(remaining_train[remaining_train.gt(0)].index)
            | set(remaining_test[remaining_test.gt(0)].index)
        )
        raise RuntimeError(f"Missing values remain after imputation: {unresolved}")

    summary = pd.DataFrame(
        {
            "feature": train.columns,
            "train_missing_before": before_train.reindex(train.columns).values,
            "test_missing_before": before_test.reindex(train.columns).values,
            "train_missing_after": remaining_train.reindex(train.columns).values,
            "test_missing_after": remaining_test.reindex(train.columns).values,
            "strategy": [parameters.get(c, {}).get("strategy", "not_imputed") for c in train.columns],
        }
    )

    train.to_csv(TRAIN_OUTPUT, index=False)
    test.to_csv(TEST_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    with open(PARAMETERS_OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(parameters, handle, ensure_ascii=False, indent=2)

    print(f"Training set: {train.shape}; test set: {test.shape}")
    print(f"Saved: {TRAIN_OUTPUT}, {TEST_OUTPUT}")
    print(f"Saved audit files: {SUMMARY_OUTPUT}, {PARAMETERS_OUTPUT}")


if __name__ == "__main__":
    main()
