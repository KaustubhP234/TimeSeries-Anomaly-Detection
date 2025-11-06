
# src/models_if.py
"""
Isolation Forest model utilities.
"""
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X: pd.DataFrame, contamination: float = 0.01, n_estimators: int = 200,
                           model_path: str = "models/isolation_forest.joblib") -> IsolationForest:
    """
    Fit IsolationForest on X (DataFrame) and save the model.
    Returns fitted model.
    """
    iso = IsolationForest(random_state=42, n_estimators=n_estimators, contamination=contamination, max_samples='auto')
    iso.fit(X)
    joblib.dump(iso, model_path)
    logging.info("IsolationForest trained and saved to %s", model_path)
    return iso


def predict_isolation_forest(iso: IsolationForest, X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions and anomaly scores.
    Returns a DataFrame with iso_label (1 normal, -1 anomaly) and iso_score (higher -> more anomalous).
    """
    labels = iso.predict(X)
    scores = -iso.decision_function(X)  # higher scores = more anomalous
    df = pd.DataFrame(index=X.index)
    df['iso_label'] = labels
    df['iso_score'] = scores
    logging.info("IsolationForest predictions generated for %d rows", len(X))
    return df
