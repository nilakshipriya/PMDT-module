"""
*** SUPERSEDED — this file's logic is now integrated directly into
*** pmdt_unified_optimized.py (see the "5b. KALMAN-FILTER SYNC BASELINE"
*** section there: run_kalman_filter_on_group / run_kalman_filter_sync,
*** wired into run_sync_pipeline as the "kalman_filter" variant).
***
*** Kept only for reference / backward compatibility with any external
*** script that still imports it directly. New work should use the
*** integrated version in the main file, which is what stage_sync() /
*** run_sync_pipeline() actually call.

Real scalar linear Kalman-filter baseline for the physics-residual
voltage-synchronization task (Section III-F of the manuscript / RED FLAG A
of the audit). The paper reports a Kalman-filter row in Table IV
(MAE=0.00017 V, RMSE=0.00259 V) but the supplied
`pmdt_unified_optimized.py` contains no Kalman-filter code at all
(grep for "kalman" / "Kalman" turns up nothing in the source). This module
supplies an actual, runnable implementation of the filter described in the
manuscript text, so Table IV's KF row is regenerated from real code
rather than reported as an unverified number.

Filter specification (matches the manuscript's prose description):
  - State: scalar SoC estimate (percent), x.
  - Process model: Coulomb counting against the FIXED assumed capacity
    the onboard BMS uses (CAP_EST_AH = 50.0 Ah, i.e. the same model that
    produces `Battery_SoC` / SoC_est in the simulator):
        x_pred = x_prev - I * dt / (CAP_EST_AH * 3600) * 100
        P_pred = P_prev + Q_proc
  - Measurement model: the SAME equivalent-circuit equation the plant
    simulator itself uses, linear in SoC (no Jacobian needed):
        z = OCV(x) - I * r_int,   OCV(x) = 44.0 + 0.10 * x
        H = dOCV/dx = 0.10
  - Standard scalar KF predict/update recursion.
  - Warm start: at the first row of each held-out temporal test segment
    (per config_id), x is initialized to that segment's own
    `Battery_SoC` value (i.e. exactly the assumed-capacity BMS estimate
    at that instant) -- this is what "warm-started from the ground-truth
    BMS SoC" means operationally: the filter is not asked to also solve
    a cold-start initial-condition problem, only to track drift from
    that point forward.
  - Prediction used for scoring at each step is the *a priori* measurement
    prediction h(x_pred) (i.e. forecast before seeing z at this step),
    exactly analogous to how the physics-only baseline and the LSTM
    variants score a prediction against the realized voltage at that
    step -- not the smoothed a posteriori estimate.
"""

import numpy as np


def battery_ocv(soc_pct):
    return 44.0 + 0.10 * soc_pct


def run_kalman_filter_on_group(
    times, currents, r_int, battery_soc_init_series, v_true,
    cap_est_ah=50.0,
    q_proc=1e-4,
    r_meas=1e-6,
):
    """
    Run the scalar KF over one config's temporal test segment.

    Parameters
    ----------
    times : (n,) array, seconds (used only to get per-step dt)
    currents : (n,) array, Battery_Current (A)
    r_int : float, this design's true internal resistance (Ohm) --
        NOTE: using the true r_int here (not an assumed value) is
        identical in spirit to what the analytic physics-only baseline
        in the paper's own code does (V_physics uses df["r_int"], the
        true design value), so the KF is being given the same
        information privilege the paper's own physics-only baseline
        already has. We flag this explicitly since it is one of several
        reasons the KF is not a "fair" real-world baseline (see the
        audit's Red-Flag-I discussion) -- it is reproduced here exactly
        as the paper's plain-language description implies, not
        strengthened or weakened relative to that description.
    battery_soc_init_series : float, Battery_SoC (SoC_est) value at the
        first row of this segment -- the warm-start state.
    v_true : (n,) array, true Battery_Voltage
    cap_est_ah : float, fixed assumed capacity used by the process model
    q_proc : float, process noise variance (SoC^2 units)
    r_meas : float, measurement noise variance (V^2 units)

    Returns
    -------
    v_pred : (n,) array of a-priori predicted voltages, one per row
             (row 0's "prediction" is simply h(x0), i.e. no propagation
             has happened yet -- included for alignment, excluded from
             scoring by the caller if desired).
    """
    n = len(times)
    x = float(battery_soc_init_series)
    P = 1e-3

    v_pred = np.zeros(n, dtype=np.float64)
    H = 0.10  # dOCV/dSoC

    for k in range(n):
        if k == 0:
            dt = 0.0
        else:
            dt = float(times[k] - times[k - 1])

        I = float(currents[k])

        # ---- predict ----
        if k > 0:
            x_pred = x - (I * dt) / (cap_est_ah * 3600.0) * 100.0
            P_pred = P + q_proc
        else:
            x_pred = x
            P_pred = P

        # a-priori measurement prediction (this is what gets scored)
        z_pred = battery_ocv(x_pred) - I * r_int
        v_pred[k] = z_pred

        # ---- update using the TRUE measured voltage ----
        z = float(v_true[k])
        innov = z - z_pred
        S = H * H * P_pred + r_meas
        K = (P_pred * H) / S if S > 0 else 0.0

        x = x_pred + K * innov
        P = (1.0 - K * H) * P_pred

    return v_pred


def run_kalman_filter_sync(test_df, seq_len, cap_est_ah=50.0,
                            q_proc=1e-4, r_meas=1e-6,
                            target_col="Battery_Voltage",
                            soc_col="Battery_SoC",
                            current_col="Battery_Current",
                            time_col="Time"):
    """
    Runs the KF independently per config_id (each config_id's rows in
    `test_df` are one physically-disjoint held-out temporal segment --
    see `sync_temporal_split_per_config`), warm-starting at that
    segment's first row, and returns predictions aligned to the SAME
    evaluation rows every other sync variant uses: rows [seq_len:] of
    each group, i.e. one prediction per row after the first `seq_len`
    rows (kept only for a fair index-aligned comparison against the
    LSTM/physics/persistence variants, which all "spend" the first
    seq_len rows of each segment as history before making their first
    prediction).
    """
    y_true_all, y_pred_all, cid_all = [], [], []

    for cfg_id, g in test_df.groupby("config_id", sort=False):
        g = g.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        n = len(g)
        if n <= seq_len:
            continue

        r_int = float(g["r_int"].iloc[0]) if "r_int" in g.columns else 0.08

        v_pred = run_kalman_filter_on_group(
            g[time_col].to_numpy(),
            g[current_col].to_numpy(),
            r_int,
            g[soc_col].iloc[0],
            g[target_col].to_numpy(),
            cap_est_ah=cap_est_ah,
            q_proc=q_proc,
            r_meas=r_meas,
        )

        y_true_all.append(g[target_col].to_numpy()[seq_len:])
        y_pred_all.append(v_pred[seq_len:])
        cid_all.append(np.full(n - seq_len, cfg_id))

    if not y_true_all:
        return np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=np.int64)

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(cid_all),
    )
