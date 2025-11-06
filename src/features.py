# src/features.py
"""
Feature engineering utilities: rolling stats, lag features, time features, scaling.
"""
import logging
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def create_features(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Create features:
      - raw sensor columns
      - rolling mean/std at windows 10 and 60
      - lag features (1,5)
      - time-of-day features if datetime index
    Returns feature DataFrame aligned to df.index and with NaNs dropped.
    """
    X = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        X[col] = df[col]
        X[f"""{col}_roll_mean_10"""] = df[col].rolling(window=10, min_periods=1).mean()
        X[f"""{col}_roll_std_10"""] = df[col].rolling(window=10, min_periods=1).std().fillna(0)
        X[f"""{col}_roll_mean_60"""] = df[col].rolling(window=60, min_periods=1).mean()
        X[f"""{col}_roll_std_60"""] = df[col].rolling(window=60, min_periods=1).std().fillna(0)
        X[f"""{col}_lag_1"""] = df[col].shift(1).fillna(method='bfill')
        X[f"""{col}_lag_5"""] = df[col].shift(5).fillna(method='bfill')
    if isinstance(df.index, pd.DatetimeIndex):
        X['hour'] = df.index.hour
        X['minute'] = df.index.minute
    X = X.dropna()
    logging.info("Created features shape: %s", X.shape)
    return X


def scale_features(X: pd.DataFrame, scaler_path: str = "models/scaler.joblib") -> (pd.DataFrame, StandardScaler):
    """
    Fit StandardScaler to X and save scaler to scaler_path.
    Returns scaled DataFrame and scaler object.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Xs = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    joblib.dump(scaler, scaler_path)
    logging.info("Features scaled and scaler saved to %s", scaler_path)
    return Xs, scaler
