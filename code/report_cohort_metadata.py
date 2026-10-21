"""Aggregate baseline dates/site coverage from saved matrices; export no participant rows."""

from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    paths = [ROOT / f"PPMI_8_{split}.csv" for split in ("train", "test")]
    train, test = [pd.read_csv(p) for p in paths]
    cohort = pd.concat([train, test], ignore_index=True)
    if len(cohort) != 1665 or cohort.PATNO.duplicated().any():
        raise RuntimeError("Unexpected cohort or duplicate participants")
    dates = pd.to_datetime(cohort.visit_date, format="%m/%Y", errors="raise")
    counts = cohort.SITE.value_counts()
    summary = {
        "n_participants": len(cohort),
        "baseline_clinical_visit_months": {
            "first": dates.min().strftime("%Y-%m"),
            "last": dates.max().strftime("%Y-%m"),
            "missing": int(dates.isna().sum()),
            "precision": "month/year; individual visit days unavailable",
        },
        "sites": {
            "n_blinded_codes": int(counts.size),
            "missing": int(cohort.SITE.isna().sum()),
            "participants_per_code_min": int(counts.min()),
            "participants_per_code_median": float(counts.median()),
            "participants_per_code_quartiles": counts.quantile([0.25, 0.75]).tolist(),
            "participants_per_code_max": int(counts.max()),
            "train_codes": int(train.SITE.nunique()),
            "test_codes": int(test.SITE.nunique()),
            "shared_codes": len(set(train.SITE) & set(test.SITE)),
            "locations": "Unavailable: curated SITE is a blinded ID; no site-location mapping supplied",
        },
        "subgroups": {
            str(k): int(v) for k, v in cohort.subgroup.value_counts().items()
        },
        "sources_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "dictionary_source": "PPMI_Curated_Data_Cut_Public_20260810.xlsx / Data dictionary",
        "notes": [
            "Baseline visit months describe the included cohort, not the accrual period of all PPMI participants.",
            "subgroup uses IU genetic consensus when available and enrolment indicators otherwise; absence of a gene label is not a confirmed negative assay.",
        ],
    }
    (ROOT / "cohort_reporting_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("n_participants", "baseline_clinical_visit_months", "sites")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
