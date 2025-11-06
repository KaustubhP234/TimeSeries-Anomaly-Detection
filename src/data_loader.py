
"""
Data loader utilities for the Time Series Anomaly Detection assignment.
Assumes CSV files in `data/`. Concatenates multiple CSVs into a single DataFrame.
"""
import logging
from pathlib import Path
import pandas as pd


def load_csv_timeseries(folder: str = "data") -> pd.DataFrame:
    """
    Load CSV files from `folder`, add source_file column, concatenate and return DataFrame.
    Raises FileNotFoundError if no CSVs found.
    """
    folder = Path(folder)
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {folder.resolve()}. Place dataset CSVs there.")
    dfs = []
    for f in csvs:
        logging.info(f"Loading {f.name}")
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            dfs.append(df)
        except Exception as e:
            logging.error(f"Failed to read {f.name}: {e}")
            raise
    combined = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded {len(dfs)} files, combined shape: {combined.shape}")
    return combined
