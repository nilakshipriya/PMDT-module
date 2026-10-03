"""
*** SUPERSEDED — this file's logic is now integrated directly into
*** pmdt_unified_optimized.py (see run_horizon_sweep,
*** make_sequences_multi_horizon, persistence_targets_horizon there).
***
*** Kept only for reference / backward compatibility. New work should use
*** the integrated version, which no longer needs `import
*** pmdt_unified_optimized as m` since it lives in the same module.

Real implementation of the multi-horizon forecasting sweep (manuscript
Section IV-E / Table VI, RED FLAG C of the audit). The supplied
`pmdt_unified_optimized.py` has no horizon parameter, horizon loop, or
target-shifting code anywhere (grep for "horizon" returns nothing), so
Table VI as printed in the paper cannot be regenerated from the supplied
artifact. This module adds that missing machinery: a target-shift
generalization of `make_sequences_multi` plus a small training/eval loop
across a list of horizons, fixing the input window and test designs and
only moving how far past the window the target sits, exactly as the
manuscript describes.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import pmdt_unified_optimized as m


def make_sequences_multi_horizon(df, feature_cols, target_cols, seq_len, horizon_steps):
    """
    Same contract as `make_sequences_multi`, except the target is read
    `horizon_steps` rows after the end of the window instead of exactly
    one row after it. horizon_steps=1 reproduces the original function's
    behavior exactly.
    """
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    X_chunks, y_chunks, cid_chunks = [], [], []

    for cfg_id, g in df.groupby("config_id", sort=False):
        g = g.sort_values("Time", kind="mergesort")
        n = len(g)
        if n <= seq_len + horizon_steps - 1:
            continue

        F = g[feature_cols].to_numpy(dtype=np.float32)
        T = g[target_cols].to_numpy(dtype=np.float32)

        X = m._strided_windows(F, seq_len)
        if X is None:
            continue

        # window i (0-indexed) covers rows [i, i+seq_len); its target is
        # `horizon_steps` rows after the window ends, i.e. row
        # i + seq_len + horizon_steps - 1.
        target_idx_start = seq_len + horizon_steps - 1
        y = T[target_idx_start:]

        n_use = min(X.shape[0], y.shape[0])
        if n_use <= 0:
            continue

        X_chunks.append(np.asarray(X[:n_use]))
        y_chunks.append(y[:n_use])
        cid_chunks.append(np.full(n_use, cfg_id, dtype=np.int64))

    if not X_chunks:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0, len(target_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.concatenate(X_chunks),
        np.concatenate(y_chunks),
        np.concatenate(cid_chunks),
    )


def persistence_targets_horizon(df, target_cols, seq_len, horizon_steps):
    """Persistence: predict target[t+horizon] with target value at the
    end of the input window, target[t] (i.e. last observed value),
    exactly generalizing `make_target_last_sequences` to horizon>1."""
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    last_chunks, y_chunks = [], []

    for cfg_id, g in df.groupby("config_id", sort=False):
        g = g.sort_values("Time", kind="mergesort")
        T = g[target_cols].to_numpy(dtype=np.float32)
        n = len(T)
        target_idx_start = seq_len + horizon_steps - 1
        if n <= target_idx_start:
            continue
        last_idx_start = seq_len - 1
        n_use = n - target_idx_start
        last_chunks.append(T[last_idx_start:last_idx_start + n_use])
        y_chunks.append(T[target_idx_start:target_idx_start + n_use])

    if not last_chunks:
        return np.empty((0, len(target_cols))), np.empty((0, len(target_cols)))
    return np.concatenate(last_chunks), np.concatenate(y_chunks)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def run_horizon_sweep(
    fleet_csv,
    horizons_s=(10, 30, 60, 120, 300),
    decimate=10,
    seq_len=20,
    epochs=15,
    seed=0,
    eval_tier="extrapolation",
):
    """
    Fixes the input window (seq_len at `decimate`-second effective
    sampling) and the test designs (the extrapolation tier, as in the
    main prognostic experiment), and only moves how far past the window
    the target sits, in units of horizon_steps = horizon_s / decimate.
    A fresh LSTM/GRU pair is retrained independently for each horizon
    (matching the manuscript's "retrained independently for each
    horizon" implication), single seed, `epochs` epochs -- explicitly a
    lighter-weight sweep than the 3-seed/25-epoch main results, exactly
    as flagged as a limitation in the paper text.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(fleet_csv)
    df = m.decimate_per_config(df, decimate)

    df_train = df[df.split == "train"]
    df_val = df[df.split == "val"]
    df_test = df[df.split == eval_tier]

    scaler_X = m.MinMaxScaler()
    scaler_y = m.MinMaxScaler()
    scaler_X.fit(df_train[m.PROG_INPUT_FEATURES])
    scaler_y.fit(df_train[m.PROG_TARGETS])

    def scaled(g):
        g2 = g.copy()
        g2[m.PROG_INPUT_FEATURES] = scaler_X.transform(g[m.PROG_INPUT_FEATURES].to_numpy(dtype=np.float32))
        g2[m.PROG_TARGETS] = scaler_y.transform(g[m.PROG_TARGETS].to_numpy(dtype=np.float32))
        return g2

    train_s, val_s, test_s = scaled(df_train), scaled(df_val), scaled(df_test)

    rows = []
    for h_s in horizons_s:
        horizon_steps = max(1, round(h_s / decimate))

        X_tr, y_tr, _ = make_sequences_multi_horizon(
            train_s, m.PROG_INPUT_FEATURES, m.PROG_TARGETS, seq_len, horizon_steps)
        X_va, y_va, _ = make_sequences_multi_horizon(
            val_s, m.PROG_INPUT_FEATURES, m.PROG_TARGETS, seq_len, horizon_steps)
        X_te, y_te, _ = make_sequences_multi_horizon(
            test_s, m.PROG_INPUT_FEATURES, m.PROG_TARGETS, seq_len, horizon_steps)

        # persistence + linear reg on RAW (unscaled) values for interpretability
        last_raw, y_raw = persistence_targets_horizon(
            df_test, m.PROG_TARGETS, seq_len, horizon_steps)

        for cell in ["lstm", "gru"]:
            model = m.PrognosticRNN(len(m.PROG_INPUT_FEATURES), cell=cell,
                                     output_dim=len(m.PROG_TARGETS)).to(m.device)
            m.train_prognostic_model(model, X_tr, y_tr, X_va, y_va,
                                      f"{cell}-h{h_s}", epochs=epochs,
                                      batch_size=512, patience=6)
            y_pred_scaled = m.predict_model_in_batches(model, X_te)
            y_true = scaler_y.inverse_transform(y_te)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            for idx, tname in enumerate(m.PROG_TARGETS):
                rows.append(dict(
                    method=cell.upper(), target=tname, horizon_s=h_s,
                    R2=r2(y_true[:, idx], y_pred[:, idx]),
                    MAE=float(np.mean(np.abs(y_true[:, idx] - y_pred[:, idx]))),
                    n=len(y_true),
                ))

        for idx, tname in enumerate(m.PROG_TARGETS):
            rows.append(dict(
                method="Persistence", target=tname, horizon_s=h_s,
                R2=r2(y_raw[:, idx], last_raw[:, idx]),
                MAE=float(np.mean(np.abs(y_raw[:, idx] - last_raw[:, idx]))),
                n=len(y_raw),
            ))

    return pd.DataFrame(rows)
