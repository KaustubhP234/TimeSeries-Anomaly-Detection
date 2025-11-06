# run_pipeline.py
"""
Run the full pipeline end-to-end:
 - load data
 - preprocessing (missing & outliers)
 - EDA plots
 - feature engineering & scaling
 - train Isolation Forest and predict
 - train LSTM Autoencoder on first portion assumed normal and compute reconstruction error
 - save plots and models to results/ and models/
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_csv_timeseries
from src.preprocessing import ensure_datetime_index, handle_missing, clip_outliers_iqr
from src.features import create_features, scale_features
from src.models_if import train_isolation_forest, predict_isolation_forest
from src.models_lstm import create_sequences, train_lstm_autoencoder
from src.evaluate import (plot_time_series, plot_rolling_stats, plot_correlation_heatmap,
                          plot_if_anomalies, plot_recon_error_and_thresh)

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)


def main():
    logging.info("Starting pipeline...")
    # 1. Load data
    df = load_csv_timeseries("data")

    # 2. Ensure datetime index (try common timestamp columns)
    ts_col = None
    for candidate in ["timestamp", "time", "datetime", "Date", "date"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    df = ensure_datetime_index(df, ts_col=ts_col, freq=None)  # don't resample by default

    # 3. Identify numeric sensor columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # drop helper columns if present
    if 'source_file' in numeric_cols:
        numeric_cols.remove('source_file')

    if not numeric_cols:
        raise ValueError("No numeric columns found in data for sensor readings.")

    # 4. Missing & outlier handling
    df = handle_missing(df)
    df = clip_outliers_iqr(df, numeric_cols=numeric_cols, k=3.0)

    # 5. EDA plots (first 3 sensors for overview)
    plot_time_series(df, numeric_cols[:3], out_path="results/time_series_overview.png")
    plot_rolling_stats(df[numeric_cols[0]], window=60, out_path="results/rolling_mean_std.png")
    plot_correlation_heatmap(df[numeric_cols], out_path="results/correlation_heatmap.png")

    # 6. Feature engineering
    X = create_features(df, numeric_cols)
    X_scaled, scaler = scale_features(X, scaler_path="models/scaler.joblib")

    # 7. Isolation Forest
    iso = train_isolation_forest(X_scaled, contamination=0.01, n_estimators=200, model_path="models/isolation_forest.joblib")
    if_res = predict_isolation_forest(iso, X_scaled)
    # Add raw sensor column to IF results for plotting context (use first numeric sensor)
    if 'raw_sensor' not in if_res.columns:
        if_res['raw_sensor'] = df[numeric_cols[0]].reindex(if_res.index, method='nearest')

    # Save IF results
    if_res.to_csv("results/isolation_forest_results.csv")

    # 8. LSTM Autoencoder
    # Prepare sequences: assume first portion of data is normal for training
    values = X_scaled.values
    seq_len = 60
    train_frac = 0.5
    n_train = int(len(values) * train_frac)
    if n_train <= seq_len:
        raise ValueError("Not enough data to create training sequences with the chosen seq_len and train_frac.")

    X_train_vals = values[:n_train]
    X_all_vals = values
    X_train_seq = create_sequences(X_train_vals, seq_len)
    X_all_seq = create_sequences(X_all_vals, seq_len)

    # Create index aligned to sequence ends for mapping errors back to timestamps
    seq_end_idx = X_scaled.index[seq_len - 1: seq_len - 1 + len(X_all_seq)]

    # Train LSTM autoencoder on training sequences (assumed normal)
    n_features = X_train_seq.shape[2]
    lstm_model, history = train_lstm_autoencoder(X_train_seq, seq_len, n_features, epochs=50, batch_size=64, model_path="models/lstm_autoencoder.h5")

    # Compute reconstruction errors for all sequences
    X_pred = lstm_model.predict(X_all_seq, verbose=0)
    mse_seq = ( (X_all_seq - X_pred) ** 2 ).mean(axis=(1,2))  # mse per sequence
    recon_error_series = pd.Series(mse_seq, index=seq_end_idx)
    recon_error_series.to_csv("results/lstm_reconstruction_error.csv")

    # Threshold: use 99th percentile of training reconstruction errors
    train_pred = lstm_model.predict(X_train_seq, verbose=0)
    train_mse = ( (X_train_seq - train_pred) ** 2 ).mean(axis=(1,2))
    threshold = float(np.percentile(train_mse, 99))

    # 9. Save and plot results
    # IF anomaly indices
    anomaly_idx_if = if_res[if_res['iso_label'] == -1].index
    plot_if_anomalies(df[numeric_cols[0]].reindex(if_res.index), anomaly_idx_if, out_path="results/if_anomalies_overlay.png")
    plot_recon_error_and_thresh(recon_error_series, threshold, out_path="results/lstm_recon_error.png")

    logging.info("Pipeline finished. Results saved to results/ and models/.")


if __name__ == "__main__":
    main()
