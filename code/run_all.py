"""Run the locked baseline single-visit analysis pipeline in order."""

import argparse
import os
import subprocess
import sys
import time


PIPELINE_MANIFEST = [
    "step_01_extract_data.py",
    "step_02_initial_missing_stats.py",
    "step_04_process_features.py",
    "step_05_split_data.py",
    "step_06_calculate_duration.py",
    "step_07_missing_stats.py",
    "step_08_imputation.py",
    "step_09_characteristics_stats.py",
    "step_10_correlation_analysis.py",
    "step_11_remove_columns.py",
    "step_12_lasso.py",
    "step_13_lr.py",
    "step_14_shrinkage.py",
]


def resolve_start_index(start_from: str | None, skip: int) -> int:
    if start_from is None:
        return skip
    if start_from.isdigit():
        prefix = f"step_{int(start_from):02d}_"
        matches = [i for i, script in enumerate(PIPELINE_MANIFEST) if script.startswith(prefix)]
    else:
        matches = [i for i, script in enumerate(PIPELINE_MANIFEST) if script == start_from]
    if not matches:
        raise ValueError(f"Unknown --start-from value: {start_from}")
    return matches[0] + skip


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(
        description="Run the PD-MCI baseline single-visit logistic-regression pipeline."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan only")
    parser.add_argument("--skip", type=int, default=0, help="Skip N scripts after the start point")
    parser.add_argument(
        "--start-from",
        help="Start from a step number (for example 8) or exact script filename",
    )
    args = parser.parse_args()

    missing = [script for script in PIPELINE_MANIFEST if not os.path.isfile(script)]
    if missing:
        raise FileNotFoundError(f"Missing pipeline scripts: {missing}")

    raw_data = "PPMI_Curated_Data_Cut_Public_20260810.xlsx"
    if not os.path.isfile(raw_data):
        raise FileNotFoundError(f"Missing curated input: {raw_data}")

    start = resolve_start_index(args.start_from, args.skip)
    queue = PIPELINE_MANIFEST[start:]
    if not queue:
        raise ValueError("No scripts remain after applying --start-from/--skip")

    print("=" * 72)
    print("PD-MCI baseline single-visit analysis")
    print("Model: LASSO feature selection + unpenalized logistic regression")
    print("Algorithm benchmark stage: excluded")
    print("=" * 72)
    for index, script in enumerate(queue, 1):
        print(f"[{index:02d}/{len(queue):02d}] {script}")

    if args.dry_run:
        print("\nDry run complete; no analysis scripts were executed.")
        return

    started = time.time()
    for index, script in enumerate(queue, 1):
        print(f"\n[{index:02d}/{len(queue):02d}] Running {script}", flush=True)
        step_started = time.time()
        completed = subprocess.run([sys.executable, script], cwd=script_dir)
        elapsed = time.time() - step_started
        if completed.returncode != 0:
            raise SystemExit(
                f"Pipeline stopped: {script} failed with exit code "
                f"{completed.returncode} after {elapsed:.1f}s"
            )
        print(f"[OK] {script} ({elapsed:.1f}s)", flush=True)

    print(f"\nAll {len(queue)} scripts completed in {time.time() - started:.1f}s.")


if __name__ == "__main__":
    main()
