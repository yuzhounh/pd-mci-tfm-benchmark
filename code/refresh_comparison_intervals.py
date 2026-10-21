"""Add full-precision bootstrap CI columns using saved predictions and fixed seeds."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from benchmark_evaluation import paired_bootstrap_test_comparison

CODE = Path(__file__).resolve().parent


def main():
    config = json.loads((CODE / "benchmark_config.json").read_text())
    validation = {}
    for prefix, track in [
        ("step_21_benchmark", "7feat"),
        ("step_15_benchmark", "full"),
    ]:
        predictions = pd.read_csv(CODE / f"{prefix}_predictions_{track}.csv")
        path = CODE / f"{prefix}_bootstrap_diffs_{track}.csv"
        table = pd.read_csv(path)
        y = predictions.MCI.to_numpy(int)
        max_point_error = 0.0
        for index, row in table.iterrows():
            result = paired_bootstrap_test_comparison(
                y,
                predictions.prob_LR.to_numpy(),
                predictions[f"prob_{row.Model}"].to_numpy(),
                n_bootstrap=config["protocol"]["bootstrap_samples"],
                random_state=config["protocol"]["bootstrap_seed"],
            )
            for metric in ("AUPRC", "AUROC", "Brier"):
                key = f"Delta_{metric}"
                max_point_error = max(
                    max_point_error, abs(result[key]["diff_point"] - row[key])
                )
                if abs(result[key]["p_value"] - row[f"{key}_pval"]) > 1e-12:
                    raise RuntimeError(
                        f"Bootstrap tail count changed: {track}/{row.Model}/{metric}"
                    )
                table.loc[index, f"{key}_CI_lower"] = result[key]["ci_lower"]
                table.loc[index, f"{key}_CI_upper"] = result[key]["ci_upper"]
            print(f"Full-precision intervals: {track}/{row.Model}", flush=True)
        if max_point_error > 1e-7:
            raise RuntimeError(f"Saved point estimates changed: {max_point_error}")
        table.to_csv(path, index=False)
        validation[track] = {
            "n_comparators": len(table),
            "maximum_point_estimate_difference": max_point_error,
            "bootstrap_samples": config["protocol"]["bootstrap_samples"],
            "bootstrap_seed": config["protocol"]["bootstrap_seed"],
            "all_p_values_unchanged": True,
        }
    (CODE / "comparison_interval_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
