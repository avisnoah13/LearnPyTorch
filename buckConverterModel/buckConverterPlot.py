"""
Buck converter comparison: PI-controlled vs RNN-in-loop.

Plots output voltage for three test conditions from Amaral &
Cardoso 2022, overlaying the PI-only response with the response
when the PI feedback error is computed from the trained RNN's
vO prediction rather than the true physics output.

"RNN in loop" means: physics still executes every period
(producing the true iL, vC, and vO), but the PI controller
receives the RNN's predicted vO as its error signal. If the
RNN is accurate the two curves nearly coincide; degraded
tracking reveals where the plant model breaks down.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from buckConverterSim import (
    simulate, step_on, step_off, vout,
    N_sub, Ts, Vref_nom, Vin_nom,
    Kp, Ki, D_min, D_max,
)
from buckConverterRNN import BuckRNN

# RNN input window length; must match make_dataset() seq_len.
SEQ_LEN = 50

# Filesystem path to the saved BuckRNN weight file.
MODEL_PATH = Path(__file__).parent / "buckConverterRNN.pt"


# Loads trained BuckRNN weights from MODEL_PATH and returns the model in eval mode on CPU, ready for inference. No parameters. Reads MODEL_PATH from the filesystem. Returns a BuckRNN instance with weights loaded and eval mode set.
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}.\n"
            "Run buckConverterRNN.py to train first."
        )
    model = BuckRNN()
    model.load_state_dict(
        torch.load(MODEL_PATH, weights_only=True)
    )
    model.eval()
    print(f"Loaded trained BuckRNN from {MODEL_PATH}.")
    return model


# Runs the buck converter closed-loop where the PI controller uses the RNN's vO prediction as its feedback error signal instead of the true physics output. Physics still executes every period to produce ground-truth iL, vC, and vO.
# Warm-up: the first seq_len periods use real PI (no RNN) so the
# history window is populated before predictions begin.
# After warm-up, each period t:
#   1. passes the last seq_len [D, iL, vO] steps to the RNN,
#   2. computes PI error  e = Vref - vO_pred,
#   3. runs physics with the previously-decided D,
#   4. appends (D_next, true_iL, true_vO) to the history.
#
#   model     - trained BuckRNN; called with torch.no_grad()
#   Vin_arr   - float32 array (n_periods,); supply voltage [V]
#   RLoad_arr - float32 array (n_periods,); load resistance [Ohm]
#   Vref      - output voltage set-point [V]
#   n_periods - total periods to simulate; must be >= seq_len
#   seq_len   - RNN window length; must match training (50)
#   iL0       - initial inductor current [A]
#   vC0       - initial capacitor voltage [V]
#   D0        - initial duty cycle; None defaults to Vref/Vin[0]
# Returns (D_arr, iL_arr, vC_arr, vO_arr) as float32 arrays of
# shape (n_periods,). vO_arr is the true physics output [V],
# not the RNN prediction.
def simulate_rnn_loop(
        model, Vin_arr, RLoad_arr, Vref, n_periods,
        seq_len=SEQ_LEN, iL0=0.0, vC0=0.0, D0=None):
    D_w, iL_w, vC_w, vO_w = simulate(
        float(Vin_arr[0]), float(RLoad_arr[0]),
        Vref, seq_len, iL0, vC0, D0,
    )

    D_list  = list(D_w)
    iL_list = list(iL_w)
    vC_list = list(vC_w)
    vO_list = list(vO_w)

    # Physics and PI state carried forward from warm-up.
    iL = float(iL_w[-1])
    vC = float(vC_w[-1])
    # D to be applied in the next physics step.
    D  = float(D_w[-1])
    # PI velocity form requires the previous period's error.
    e_prev = Vref - float(vO_w[-1])

    print(f"Simulating RNN-in-loop for {n_periods} periods...")

    for t in range(seq_len, n_periods):
        Vin   = float(Vin_arr[t])
        RLoad = float(RLoad_arr[t])

        # Input window: most recent seq_len [D, iL, vO] rows.
        window = np.stack([
            np.array(D_list[-seq_len:],  dtype=np.float32),
            np.array(iL_list[-seq_len:], dtype=np.float32),
            np.array(vO_list[-seq_len:], dtype=np.float32),
        ], axis=1)
        x = torch.from_numpy(window[np.newaxis])
        with torch.no_grad():
            vO_pred = model(x).item()

        # PI uses predicted vO for this period's error.
        e      = Vref - vO_pred
        D_next = float(np.clip(
            D + Kp * (e - e_prev) + Ki * Ts * e,
            D_min, D_max,
        ))
        e_prev = e

        # Physics runs with D (decided last period, not D_next).
        n_on = max(1, min(N_sub - 1, int(round(D * N_sub))))
        for _ in range(n_on):
            iL, vC = step_on(iL, vC, Vin, RLoad)
        for _ in range(N_sub - n_on):
            iL, vC = step_off(iL, vC, RLoad)
        iL = max(iL, 0.0)
        true_vO = vout(iL, vC, RLoad)

        D_list.append(D_next)
        iL_list.append(iL)
        vC_list.append(vC)
        vO_list.append(true_vO)
        D = D_next

    return (
        np.array(D_list,  dtype=np.float32),
        np.array(iL_list, dtype=np.float32),
        np.array(vC_list, dtype=np.float32),
        np.array(vO_list, dtype=np.float32),
    )


# Generates and displays a 1x3 figure comparing PI-controlled
# and RNN-in-loop vO across startup, load step, and Vin step.
# Loads the trained model internally via load_model().
# No parameters.
# No return value. Writes to the matplotlib display (blocking).
def plot_comparison():
    model = load_model()

    # Total periods per scenario; 50 ms at 100 kHz switching.
    n     = 5000
    # Period at which the load / Vin step occurs (30 ms).
    n_pre = 3000
    # Time axis in milliseconds.
    t = np.arange(n) * Ts * 1e3

    # ── Scenario 1: cold startup ──────────────────────────────
    _, _, _, vO_pi_1 = simulate(Vin_nom, 1.0, Vref_nom, n)

    Vin_1   = np.full(n, Vin_nom, dtype=np.float32)
    RLoad_1 = np.full(n, 1.0,    dtype=np.float32)
    _, _, _, vO_rnn_1 = simulate_rnn_loop(
        model, Vin_1, RLoad_1, Vref_nom, n
    )

    # ── Scenario 2: load step (1 Ω → 0.5 Ω at 30 ms) ─────────
    r_pre  = simulate(Vin_nom, 1.0, Vref_nom, n_pre)
    r_post = simulate(
        Vin_nom, 0.5, Vref_nom, n - n_pre,
        iL0=r_pre[1][-1], vC0=r_pre[2][-1],
        D0=r_pre[0][-1],
    )
    vO_pi_2 = np.concatenate([r_pre[3], r_post[3]])

    Vin_2   = np.full(n, Vin_nom, dtype=np.float32)
    RLoad_2 = np.concatenate([
        np.full(n_pre, 1.0),
        np.full(n - n_pre, 0.5),
    ]).astype(np.float32)
    _, _, _, vO_rnn_2 = simulate_rnn_loop(
        model, Vin_2, RLoad_2, Vref_nom, n
    )

    # ── Scenario 3: Vin step (19 V → 9 V at 30 ms) ────────────
    r_pre  = simulate(Vin_nom, 1.0, Vref_nom, n_pre)
    r_post = simulate(
        9.0, 1.0, Vref_nom, n - n_pre,
        iL0=r_pre[1][-1], vC0=r_pre[2][-1],
        D0=r_pre[0][-1],
    )
    vO_pi_3 = np.concatenate([r_pre[3], r_post[3]])

    Vin_3   = np.concatenate([
        np.full(n_pre, Vin_nom),
        np.full(n - n_pre, 9.0),
    ]).astype(np.float32)
    RLoad_3 = np.full(n, 1.0, dtype=np.float32)
    _, _, _, vO_rnn_3 = simulate_rnn_loop(
        model, Vin_3, RLoad_3, Vref_nom, n
    )
    print("Simulation complete; plotting results...")
    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Buck Converter: PI-controlled vs RNN-in-loop  (vO [V])",
        fontsize=13,
    )

    scenarios = [
        (
            vO_pi_1, vO_rnn_1,
            "Startup  (Vin=19V, RLoad=1Ω)",
        ),
        (
            vO_pi_2, vO_rnn_2,
            "Load step  (RLoad: 1→0.5 Ω at 30 ms)",
        ),
        (
            vO_pi_3, vO_rnn_3,
            "Vin step  (19→9 V at 30 ms)",
        ),
    ]

    for ax, (vO_pi, vO_rnn, title) in zip(axes, scenarios):
        rmse = float(np.sqrt(np.mean((vO_rnn - vO_pi) ** 2)))

        ax.plot(
            t, vO_pi,
            linewidth=0.9, color="C0", label="PI controller",
        )
        ax.plot(
            t, vO_rnn,
            linewidth=0.9, color="C1", alpha=0.85,
            label="RNN in loop",
        )
        ax.axhline(
            Vref_nom, color="r", linestyle="--",
            linewidth=0.8, label=f"Vref = {Vref_nom} V",
        )
        ax.set_title(
            f"{title}\nRMSE vs PI: {rmse:.4f} V",
            fontsize=9,
        )
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("vO [V]")
        ax.legend(fontsize=8)
        ax.set_xlim(0, t[-1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_comparison()
