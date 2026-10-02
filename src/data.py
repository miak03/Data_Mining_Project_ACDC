"""Load project configuration and raw data."""
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    """Load config.yaml, resolving relative config paths from the project root."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_data(config=None) -> pd.DataFrame:
    """Load the configured raw dataset."""
    config = config or load_config()
    data_config = config["data"]
    data_path = (PROJECT_ROOT / data_config["path"]).resolve()
    return pd.read_csv(data_path, sep=data_config.get("delimiter", ","))