"""Dataset validity and quality diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Replace configured numeric-bound violations with missing values in place."""
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def write_diagnostics(
    raw_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    output_dir: str | Path,
    validity_report: pd.DataFrame,
    id_column: str | None = None,
) -> None:
    """Write duplicate, missingness, and validity reports to the configured folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duplicate_rows = raw_data[raw_data.duplicated(keep=False)]
    if id_column in duplicate_rows.columns:
        ordered_columns = [id_column] + [
            column for column in duplicate_rows.columns if column != id_column
        ]
        duplicate_rows = duplicate_rows[ordered_columns]
    duplicate_rows.to_csv(output_dir / "duplicate_rows.csv", index=False)

    missingness = pd.DataFrame({
        "n_missing": cleaned_data.isna().sum(),
        "fraction_missing": cleaned_data.isna().mean(),
    })
    missingness[missingness["n_missing"] > 0].sort_values(
        "fraction_missing", ascending=False
    ).to_csv(output_dir / "missingness.csv")
    validity_report.to_csv(output_dir / "validity_rules.csv", index=False)