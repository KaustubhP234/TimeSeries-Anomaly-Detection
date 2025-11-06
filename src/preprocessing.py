
# src/preprocessing.py
"""
Missing values & outlier handling and basic time index utilities.
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd


def ensure_datetime_index(df: pd.DataFrame, ts_col: Optional[str] = None, freq: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure DataFrame has a datetime index.
    - If ts_col provided and exists, convert to datetime and set index.
    - Else, create a synthetic range index (careful: only for demonstration).
    If freq is provided, attempt to resample to that frequency using forward-fill.
    """
    if ts_col and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values(ts_col).set_index(ts_col)
        logging.info("Converted column %s to datetime index.", ts_col)
    else:
        # Create a synthetic index if none provided
        idx = pd.date_range(start="2020-01-01", periods=len(df), freq="S")
        df.index = idx
        logging.info("No timestamp column provided; created synthetic second-level index.")
    if freq:
        try:
            df = df.resample(freq).mean().interpolate(method='time')
            logging.info("Resampled to frequency %s", freq)
        except Exception as e:
            logging.warning("Resampling failed: %s", e)
    return df


def handle_missing(df: pd.DataFrame, interp_limit: int = 100) -> pd.DataFrame:
    """
    Interpolate small gaps with time-based interpolation up to interp_limit,
    then forward/back-fill remaining missing values.
    """
    df_interp = df.interpolate(method='time', limit=interp_limit)
    df_interp = df_interp.fillna(method='ffill').fillna(method='bfill')
    logging.info("Missing values after handling: %d", df_interp.isna().sum().sum())
    return df_interp


def clip_outliers_iqr(df: pd.DataFrame, numeric_cols=None, k: float = 3.0) -> pd.DataFrame:
    """
    Clip outliers using IQR per numeric column. Uses clip (does not drop rows).
    k: multiplier for IQR. Larger k => more conservative.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    clipped = df.copy()
    for col in numeric_cols:
        try:
            q1 = clipped[col].quantile(0.25)
            q3 = clipped[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            clipped[col] = clipped[col].clip(lower, upper)
        except Exception as e:
            logging.warning("Skipping clipping for %s: %s", col, e)
    logging.info("Completed IQR clipping with k=%s for %d columns", k, len(numeric_cols))
    return clipped
