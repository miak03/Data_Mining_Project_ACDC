"""Dataset validity and quality diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd


def flag_invalid_values(
    df: pd.DataFrame,
    rules: dict,
    id_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report numeric-bound violations, then replace those values with missing in place."""
    report_rows = []
    detail_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        below_min = numeric.notna() & (numeric < bounds["min"]) if "min" in bounds else pd.Series(False, index=numeric.index)
        above_max = numeric.notna() & (numeric > bounds["max"]) if "max" in bounds else pd.Series(False, index=numeric.index)
        violations = below_min | above_max
        report_rows.append({
            "column": column,
            "min_allowed": bounds.get("min"),
            "max_allowed": bounds.get("max"),
            "below_min": int(below_min.sum()),
            "above_max": int(above_max.sum()),
            "violations": int(violations.sum()),
        })

        for condition, mask in (("below_min", below_min), ("above_max", above_max)):
            if not mask.any():
                continue
            selected_columns = [id_column, column] if id_column in df.columns else [column]
            details = df.loc[mask, selected_columns].copy()
            if id_column not in df.columns:
                details.insert(0, "dataframe_index", df.index[mask])
            details = details.rename(columns={column: "original_value"})
            details["column"] = column
            details["condition"] = condition
            details["min_allowed"] = bounds.get("min")
            details["max_allowed"] = bounds.get("max")
            detail_columns = [
                name for name in (id_column, "dataframe_index") if name and name in details.columns
            ]
            detail_rows.append(details[
                detail_columns
                + ["column", "original_value", "condition", "min_allowed", "max_allowed"]
            ])
        df.loc[violations, column] = np.nan

    summary = pd.DataFrame(report_rows)
    detail_columns = [id_column] if id_column else ["dataframe_index"]
    detail_columns += ["column", "original_value", "condition", "min_allowed", "max_allowed"]
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(columns=detail_columns)
    return summary, details


def find_date_order_violations(
    df: pd.DataFrame,
    rules: list,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Report end-before-start date pairs without changing source values."""
    violation_reports = []
    empty_columns = [id_column] if id_column else []
    for rule in rules:
        start_column = rule["start_column"]
        end_column = rule["end_column"]
        empty_columns.extend([start_column, end_column])
        if start_column not in df.columns or end_column not in df.columns:
            continue

        start_dates = pd.to_datetime(df[start_column], errors="coerce")
        end_dates = pd.to_datetime(df[end_column], errors="coerce")
        ignored_end_dates = pd.to_datetime(rule.get("ignore_end_values", []), errors="coerce")
        invalid_order = (
            start_dates.notna()
            & end_dates.notna()
            & end_dates.lt(start_dates)
            & ~end_dates.isin(ignored_end_dates)
        )

        report_columns = [
            column for column in (id_column, start_column, end_column)
            if column and column in df.columns
        ]
        violations = df.loc[invalid_order, report_columns].copy()
        violations["issue"] = "end_before_start"
        violation_reports.append(violations)

    if violation_reports:
        return pd.concat(violation_reports, ignore_index=True)
    return pd.DataFrame(columns=[*dict.fromkeys(empty_columns), "issue"])


def write_diagnostics(
    raw_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    output_dir: str | Path,
    validity_report: pd.DataFrame,
    invalid_value_details: pd.DataFrame,
    date_order_violations: pd.DataFrame,
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
    validity_report.to_csv(output_dir / "invalid_value_summary.csv", index=False)
    invalid_value_details.to_csv(output_dir / "invalid_values.csv", index=False)
    date_order_violations.to_csv(output_dir / "date_order_violations.csv", index=False)
    type_report = pd.DataFrame({
        "raw_dtype": raw_data.dtypes.astype(str),
        "cleaned_dtype": cleaned_data.dtypes.astype(str),
        "missing_after_cleaning": cleaned_data.isna().sum(),
    })
    type_report.index.name = "column"
    type_report.to_csv(output_dir / "data_types.csv")