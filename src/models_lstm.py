# src/models_lstm.py
"""
LSTM autoencoder utilities: sequence creation, model build/train, reconstruction errors.
"""
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from typing import Tuple


def create_sequences(values: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Convert 2D array (n_rows, n_features) to sequences of shape (n_sequences, seq_len, n_features).
    """
    sequences = []
    for i in range(len(values) - seq_len + 1):
        sequences.append(values[i:i + seq_len])
    return np.array(sequences)


def build_lstm_autoencoder(seq_len: int, n_features: int) -> tf.keras.Model:
    """
    Build a simple LSTM autoencoder.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(64, activation='tanh', return_sequences=False),
        layers.RepeatVector(seq_len),
        layers.LSTM(64, activation='tanh', return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    logging.info("Built LSTM autoencoder (seq_len=%d, n_features=%d)", seq_len, n_features)
    return model


def train_lstm_autoencoder(X_train_seq: np.ndarray, seq_len: int, n_features: int,
                           epochs: int = 50, batch_size: int = 64, model_path: str = "models/lstm_autoencoder.h5") -> Tuple[tf.keras.Model, dict]:
    """
    Train LSTM autoencoder on provided sequence array (assumes normal data).
    Returns the model and training history dict.
    """
    model = build_lstm_autoencoder(seq_len, n_features)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_seq, X_train_seq,
        epochs=epochs, batch_size=batch_size,
        validation_split=0.1, callbacks=[es], verbose=1
    )
    model.save(model_path)
    logging.info("LSTM autoencoder trained and saved to %s", model_path)
    return model, history.history


def compute_reconstruction_errors(model: tf.keras.Model, X_seq: np.ndarray, index_for_seq: pd.DatetimeIndex) -> pd.Series:
    """
    Compute MSE per sequence and map result to the last timestamp of each sequence (index_for_seq).
    index_for_seq should be the original X.index corresponding to sequence ends.
    """
    X_pred = model.predict(X_seq, verbose=0)
    mse_seq = np.mean(np.mean((X_seq - X_pred) ** 2, axis=2), axis=1)
    # Map sequence errors to the last timestamp of each sequence
    seq_end_idx = np.arange(len(index_for_seq))  # index_for_seq should be provided accordingly
    series = pd.Series(mse_seq, index=index_for_seq)
    logging.info("Computed reconstruction errors for %d sequences", len(mse_seq))
    return series
