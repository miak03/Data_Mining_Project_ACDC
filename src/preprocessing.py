
"""Configured cleaning steps for the ACDC dataset."""
import pandas as pd

from src.diagnostics import flag_invalid_values, write_diagnostics


def canonicalize_categories(df: pd.DataFrame, category_maps: dict, placeholder_tokens: list) -> pd.DataFrame:
    """Normalize configured categories and convert placeholders to missing values."""
    out = df.copy()
    placeholder_lower = {str(token).strip().lower() for token in placeholder_tokens}
    for column, mapping in category_maps.items():
        if column not in out.columns:
            continue
        cleaned = out[column].astype("string").str.strip()
        lowered = cleaned.str.lower()
        canonicalized = lowered.map(mapping).fillna(cleaned)
        out[column] = canonicalized.mask(lowered.isin(placeholder_lower), pd.NA)
    return out


def canonicalize_boolean_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert common boolean spellings to pandas nullable booleans."""
    out = df.copy()
    value_map = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
        "1": True,
        "0": False,
    }
    for column in columns:
        if column not in out.columns:
            continue
        cleaned = out[column].astype("string").str.strip().str.lower()
        out[column] = cleaned.map(value_map).astype("boolean")
    return out


def clean_dataset(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply configured cleaning and return the cleaned data and validity report."""
    diagnostics_config = config["diagnostics"]
    out = df.drop_duplicates(keep="first").reset_index(drop=True).copy()
    validity_report = flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
    out = canonicalize_categories(
        out,
        diagnostics_config.get("canonical_categories", {}),
        diagnostics_config.get("placeholder_tokens", []),
    )
    out = canonicalize_boolean_columns(out, diagnostics_config.get("boolean_columns", []))
    return out, validity_report

