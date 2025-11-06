# src/evaluate.py
"""
Evaluation and plotting utilities: EDA plots, anomaly overlays, saving to results/.
"""
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def plot_time_series(df: pd.DataFrame, cols: list, out_path: str = "results/time_series_overview.png"):
    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(14, 4))
    for col in cols:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.title("Sensor readings (overview)")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info("Saved time series overview to %s", out_path)


def plot_rolling_stats(series: pd.Series, window: int = 60, out_path: str = "results/rolling_mean_std.png"):
    Path("results").mkdir(exist_ok=True)
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std = series.rolling(window=window, min_periods=1).std().fillna(0)
    plt.figure(figsize=(14, 4))
    plt.plot(series.index, series.values, alpha=0.5, label='raw')
    plt.plot(roll_mean.index, roll_mean.values, label=f'rolling mean ({window})')
    plt.fill_between(roll_mean.index, roll_mean - roll_std, roll_mean + roll_std, alpha=0.2)
    plt.legend()
    plt.title("Rolling mean & std")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info("Saved rolling stats to %s", out_path)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: str = "results/correlation_heatmap.png"):
    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation heatmap")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info("Saved correlation heatmap to %s", out_path)


def plot_if_anomalies(sensor_series: pd.Series, anomaly_index: pd.DatetimeIndex, out_path: str = "results/if_anomalies_overlay.png"):
    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(14, 4))
    plt.plot(sensor_series.index, sensor_series.values, label='sensor')
    if len(anomaly_index):
        plt.scatter(anomaly_index, sensor_series.loc[anomaly_index], color='red', s=10, label='IsolationForest anomalies')
    plt.legend()
    plt.title("Isolation Forest anomalies overlay")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info("Saved IF anomalies overlay to %s", out_path)


def plot_recon_error_and_thresh(error_series: pd.Series, threshold: float, out_path: str = "results/lstm_recon_error.png"):
    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(14, 4))
    plt.plot(error_series.index, error_series.values, label='reconstruction error')
    plt.hlines(threshold, error_series.index[0], error_series.index[-1], colors='r', linestyles='dashed', label='threshold')
    anomalies = error_series[error_series > threshold]
    if not anomalies.empty:
        plt.scatter(anomalies.index, anomalies.values, color='red', s=10, label='LSTM anomalies')
    plt.legend()
    plt.title("LSTM reconstruction error and anomalies")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info("Saved LSTM reconstruction error plot to %s", out_path)
