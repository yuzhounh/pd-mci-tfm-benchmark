"""Create the locked stratified split for one baseline row per participant."""

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "PPMI_4_features.csv"
TRAIN_OUTPUT = "PPMI_5_train.csv"
TEST_OUTPUT = "PPMI_5_test.csv"
TEST_SIZE = 0.30
RANDOM_STATE = 42
EXPECTED_SPLIT = (1165, 1165, 500, 500)


def validate_single_visit(data: pd.DataFrame) -> None:
    if not data["EVENT_ID"].eq("BL").all():
        unexpected = sorted(data.loc[data["EVENT_ID"].ne("BL"), "EVENT_ID"].unique())
        raise RuntimeError(f"Expected baseline-only data; found EVENT_ID values: {unexpected}")
    duplicates = data.loc[data["PATNO"].duplicated(keep=False), "PATNO"].unique()
    if len(duplicates):
        raise RuntimeError(
            "Single-visit invariant failed; duplicate PATNO values found: "
            f"{duplicates[:10].tolist()}"
        )


def describe(name: str, data: pd.DataFrame) -> None:
    print(f"{name}: {len(data)} rows / {data['PATNO'].nunique()} participants")
    print(f"  MCI counts: {data['MCI'].value_counts().sort_index().to_dict()}")
    print(f"  MCI prevalence: {data['MCI'].mean():.4f}")


def main() -> None:
    print("=== Step 05: Baseline Single-Visit Train/Test Split ===")
    data = pd.read_csv(INPUT_FILE)
    validate_single_visit(data)

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        stratify=data["MCI"],
        random_state=RANDOM_STATE,
    )
    train_data = train_data.sort_values("PATNO").reset_index(drop=True)
    test_data = test_data.sort_values("PATNO").reset_index(drop=True)

    overlap = set(train_data["PATNO"]) & set(test_data["PATNO"])
    if overlap:
        raise RuntimeError(f"Train/test participant overlap detected: {sorted(overlap)[:10]}")

    observed = (
        len(train_data),
        train_data["PATNO"].nunique(),
        len(test_data),
        test_data["PATNO"].nunique(),
    )
    if observed != EXPECTED_SPLIT:
        raise RuntimeError(f"Locked split changed: expected {EXPECTED_SPLIT}, got {observed}")

    describe("Training set", train_data)
    describe("Held-out test set", test_data)
    train_data.to_csv(TRAIN_OUTPUT, index=False)
    test_data.to_csv(TEST_OUTPUT, index=False)
    print(f"Saved: {TRAIN_OUTPUT}, {TEST_OUTPUT}")


if __name__ == "__main__":
    main()
