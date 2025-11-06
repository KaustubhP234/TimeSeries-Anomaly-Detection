# src/generate_data.py
"""
Generate synthetic multivariate time series data with embedded anomalies
for IoT sensor anomaly detection demonstration.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random

def generate_synthetic_sensor_data(
    n_samples: int = 10000,
    anomaly_fraction: float = 0.02,
    n_sensors: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic IoT sensor data with injected anomalies.
    - Normal data: sinusoidal + trend + Gaussian noise.
    - Anomalies: random spikes, drifts, or drops.
    """
    np.random.seed(seed)
    random.seed(seed)
    time_index = pd.date_range(start="2024-01-01", periods=n_samples, freq="S")

    data = {}
    for i in range(n_sensors):
        # Base signal: sine wave + trend + random noise
        trend = 0.0001 * np.arange(n_samples)
        base_signal = np.sin(0.02 * np.arange(n_samples)) + trend
        noise = np.random.normal(0, 0.05, n_samples)
        sensor = base_signal + noise

        # Inject anomalies
        n_anomalies = int(anomaly_fraction * n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

        for idx in anomaly_indices:
            anomaly_type = random.choice(["spike", "drop", "drift"])
            if anomaly_type == "spike":
                sensor[idx:idx + 3] += np.random.uniform(3, 5)  # sudden spike
            elif anomaly_type == "drop":
                sensor[idx:idx + 3] -= np.random.uniform(3, 5)  # sudden drop
            elif anomaly_type == "drift":
                drift_length = np.random.randint(20, 50)
                drift_value = np.random.uniform(1, 2)
                end_idx = min(idx + drift_length, n_samples)
                sensor[idx:end_idx] += np.linspace(0, drift_value, end_idx - idx)

        data[f"sensor_{i+1}"] = sensor

    df = pd.DataFrame(data, index=time_index)
    df.index.name = "timestamp"

    # Optional: add an "anomaly_flag" for evaluation (1=anomaly, 0=normal)
    df["anomaly_flag"] = 0
    for i in range(n_sensors):
        sensor_col = f"sensor_{i+1}"
        anomalies = (np.abs(df[sensor_col] - df[sensor_col].rolling(100, min_periods=1).mean()) > 3*df[sensor_col].rolling(100, min_periods=1).std())
        df.loc[anomalies, "anomaly_flag"] = 1

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/sensor_data.csv")
    print(f"✅ Synthetic data generated: {df.shape[0]} rows, {df.shape[1]-1} sensors.")
    print("File saved at: data/sensor_data.csv")
    return df


if __name__ == "__main__":
    generate_synthetic_sensor_data()
