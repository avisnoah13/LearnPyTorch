"""
Buck converter closed-loop PI simulation.
Reference: Amaral & Cardoso 2022, Signals 3, 313-325.
doi:10.3390/signals3020020

State equations (K1 = C*(1 + ESR/RLoad)):
  dvC/dt = iL/K1 - vC/(RLoad*K1)          (both stages)
  diL/dt = Vin/L + iL*KA_on  + vC*KB      (Stage I)
    KA_on  = -(RS+RL)/L - ESR*C/(K1*L)
  diL/dt = -Vd/L + iL*KA_off + vC*KB      (Stage II)
    KA_off = -(Rd+RL)/L - ESR*C/(K1*L)
  KB = -C/(K1*L),  vO = vC + ESR*(iL - vC/RLoad)

PI controller (velocity form, once per switching period):
  e[k] = Vref - vO[k]
  D[k] = clip(
    D[k-1] + Kp*(e[k]-e[k-1]) + Ki*Ts*e[k], D_min, D_max
  )

ML dataset layout (many-to-one RNN, see CLAUDE.md):
  X[i] shape (seq_len, 3): [D, iL, vO] sliding window
  y[i] shape (1,):         vO one step past the window end
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# ── Circuit parameters (Table 1, Amaral & Cardoso 2022) ────────────
Vin_nom  = 19.0    # V    nominal input voltage
Vref_nom =  5.0    # V    nominal output set-point (D_nom ~= 0.26)
L        = 200e-6  # H    inductance
RL       =  0.05   # Ohm  inductor winding resistance
C        = 220e-6  # F    capacitance (Alcap)
ESR      =  0.2    # Ohm  cap ESR at 1 kHz (Alcap, paper sec. 3)
RS       =  0.05   # Ohm  MOSFET on-resistance (assumed)
Rd       =  0.02   # Ohm  diode series resistance (assumed)
Vd       =  0.4    # V    diode forward voltage (assumed)
fs       = 100e3   # Hz   switching frequency
Ts       = 1.0 / fs  # s  switching period = 10 us

# ── PI controller ───────────────────────────────────────────────────
# Open-loop DC gain ~= Ki*Ts*Vin ~= 500*10e-6*19 ~= 0.095 -> stable.
Kp    = 0.05
Ki    = 500.0
D_min = 0.05   # prevent full-off (DCM guard)
D_max = 0.95   # prevent full-on (saturation guard)

# ── Euler integration ───────────────────────────────────────────────
N_sub = 100          # sub-steps per switching period
dt    = Ts / N_sub   # 0.1 us per sub-step


# Integrates the Stage I circuit equations (switch ON, diode
# reverse-biased) forward by one sub-step dt using forward Euler.
# During Stage I, Vin charges the inductor and current flows into
# the capacitor and load. Called N_sub * D times per period.
#   iL    - inductor current at the start of this sub-step [A]
#   vC    - capacitor voltage at the start of this sub-step [V]
#   Vin   - DC input supply voltage [V]
#   RLoad - load resistance [Ohm]
# Returns (iL_next, vC_next): updated inductor current [A] and
# capacitor voltage [V] after one sub-step dt has elapsed.
def step_on(iL, vC, Vin, RLoad):
    k1  = C * (1.0 + ESR / RLoad)
    KA  = -(RS + RL) / L - ESR * C / (k1 * L)
    KB  = -C / (k1 * L)
    dvC = iL / k1 - vC / (RLoad * k1)
    diL = Vin / L + iL * KA + vC * KB
    return iL + dt * diL, vC + dt * dvC


# Integrates the Stage II circuit equations (switch OFF, diode
# freewheeling) forward by one sub-step dt using forward Euler.
# During Stage II, the inductor releases stored energy through
# the freewheeling diode into the capacitor and load; Vin is
# disconnected. Called N_sub * (1 - D) times per period.
#   iL    - inductor current at the start of this sub-step [A]
#   vC    - capacitor voltage at the start of this sub-step [V]
#   RLoad - load resistance [Ohm]
# Returns (iL_next, vC_next): updated inductor current [A] and
# capacitor voltage [V] after one sub-step dt has elapsed.
def step_off(iL, vC, RLoad):
    k1  = C * (1.0 + ESR / RLoad)
    KA  = -(Rd + RL) / L - ESR * C / (k1 * L)
    KB  = -C / (k1 * L)
    dvC = iL / k1 - vC / (RLoad * k1)
    diL = -Vd / L + iL * KA + vC * KB
    return iL + dt * diL, vC + dt * dvC


# Computes the terminal output voltage, which differs from vC
# because the capacitor's ESR adds a voltage drop proportional
# to the current flowing through it. vC is the ideal stored
# voltage; vO is what a voltmeter at the output would read.
#   iL    - inductor current at this instant [A]
#   vC    - pure capacitive state variable [V]
#   RLoad - load resistance [Ohm]; determines how much of iL
#           flows into the load vs. into the capacitor
# Returns vO: terminal output voltage including ESR drop [V].
def vout(iL, vC, RLoad):
    return vC + ESR * (iL - vC / RLoad)


# Simulates n_periods complete switching cycles of the closed-loop
# buck converter starting from the given initial conditions. Each
# period runs N_sub Euler sub-steps split between Stage I and
# Stage II according to D, then updates D with the PI controller.
# One sample is recorded per period (end-of-period values),
# capturing averaged dynamics without sub-step ripple.
# To chain episodes, pass the final iL, vC, D from one call as
# iL0, vC0, D0 of the next (e.g. to simulate a load step).
#   Vin      - input supply voltage for this episode [V]
#   RLoad    - load resistance for this episode [Ohm]
#   Vref     - output voltage set-point the PI regulates to [V]
#   n_periods - number of switching periods (samples) to run
#   iL0      - initial inductor current; 0.0 for cold start [A]
#   vC0      - initial capacitor voltage; 0.0 for cold start [V]
#   D0       - initial duty cycle; defaults to Vref/Vin
# Returns four float32 arrays of shape (n_periods,):
#   D_arr  - duty cycle commanded each period (0 to 1)
#   iL_arr - inductor current at end of each period [A]
#   vC_arr - capacitor voltage at end of each period [V];
#            pass as vC0 when chaining episodes
#   vO_arr - terminal output voltage at end of each period [V]
def simulate(Vin, RLoad, Vref, n_periods, iL0=0.0, vC0=0.0, D0=None):
    if D0 is None:
        D0 = float(np.clip(Vref / Vin, D_min, D_max))

    D_arr  = np.empty(n_periods, dtype=np.float32)
    iL_arr = np.empty(n_periods, dtype=np.float32)
    vC_arr = np.empty(n_periods, dtype=np.float32)
    vO_arr = np.empty(n_periods, dtype=np.float32)

    iL, vC, D, e_prev = float(iL0), float(vC0), float(D0), 0.0

    for k in range(n_periods):
        n_on = max(1, min(N_sub - 1, int(round(D * N_sub))))

        for _ in range(n_on):
            iL, vC = step_on(iL, vC, Vin, RLoad)
        for _ in range(N_sub - n_on):
            iL, vC = step_off(iL, vC, RLoad)

        iL = max(iL, 0.0)  # diode blocks reverse current (CCM guard)

        vO = vout(iL, vC, RLoad)
        e  = Vref - vO
        D  = float(np.clip(
            D + Kp * (e - e_prev) + Ki * Ts * e, D_min, D_max
        ))
        e_prev = e

        D_arr[k], iL_arr[k], vC_arr[k], vO_arr[k] = D, iL, vC, vO

    return D_arr, iL_arr, vC_arr, vO_arr


# Simulates three operating scenarios and plots vO, iL, and D
# over time in a 3-row x 3-column figure. Scenarios match the
# paper's validation conditions (Figs. 12-14): startup transient,
# load current step, and input voltage step. Each column is one
# scenario; each row is one signal. Blocks until the window
# is closed.
# No inputs. No return value.
def plot_waveforms():
    n = 5000
    t = np.arange(n) * Ts * 1e3  # time axis in ms

    # scenario 1: cold startup at nominal conditions
    D1, iL1, _, vO1 = simulate(Vin_nom, 1.0, Vref_nom, n)

    # scenario 2: load step (1 Ohm -> 0.5 Ohm at 30 ms,
    # doubling the output current from 5 A to 10 A)
    n_pre  = 3000
    r_pre  = simulate(Vin_nom, 1.0, Vref_nom, n_pre)
    r_post = simulate(
        Vin_nom, 0.5, Vref_nom, n - n_pre,
        iL0=r_pre[1][-1], vC0=r_pre[2][-1],
        D0=r_pre[0][-1],
    )
    D2  = np.concatenate([r_pre[0], r_post[0]])
    iL2 = np.concatenate([r_pre[1], r_post[1]])
    vO2 = np.concatenate([r_pre[3], r_post[3]])

    # scenario 3: input voltage step (19 V -> 9 V at 30 ms)
    r_pre  = simulate(Vin_nom, 1.0, Vref_nom, n_pre)
    r_post = simulate(
        9.0, 1.0, Vref_nom, n - n_pre,
        iL0=r_pre[1][-1], vC0=r_pre[2][-1],
        D0=r_pre[0][-1],
    )
    D3  = np.concatenate([r_pre[0], r_post[0]])
    iL3 = np.concatenate([r_pre[1], r_post[1]])
    vO3 = np.concatenate([r_pre[3], r_post[3]])

    fig, axes = plt.subplots(3, 3, figsize=(14, 7), sharex=True)
    fig.suptitle(
        "Buck Converter Waveforms (Amaral & Cardoso 2022 conditions)",
        fontsize=12,
    )

    cols = [
        (D1, iL1, vO1, "Startup  (Vin=19V, RLoad=1Ohm, Vref=5V)"),
        (D2, iL2, vO2, "Load step  (RLoad: 1->0.5 Ohm at 30 ms)"),
        (D3, iL3, vO3, "Vin step  (19->9 V at 30 ms)"),
    ]

    for col, (D, iL, vO, title) in enumerate(cols):
        axes[0][col].set_title(title, fontsize=9)
        axes[0][col].plot(t, vO, linewidth=0.8)
        axes[0][col].axhline(
            Vref_nom, color="r", linestyle="--",
            linewidth=0.8, label="Vref",
        )
        axes[0][col].set_ylabel("vO [V]")
        axes[0][col].legend(fontsize=7)

        axes[1][col].plot(t, iL, linewidth=0.8, color="C1")
        axes[1][col].set_ylabel("iL [A]")

        axes[2][col].plot(t, D, linewidth=0.8, color="C2")
        axes[2][col].set_ylabel("D")
        axes[2][col].set_xlabel("Time [ms]")
        axes[2][col].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


# Extracts overlapping sliding windows from one simulation run
# for use as RNN training samples. Each window of length seq_len
# becomes one input sequence X; the output voltage one step past
# the window becomes the target y. A stride > 1 reduces redundancy
# in long settled regions of the time series.
#   D_arr   - duty cycle series from simulate(), shape (T,)
#   iL_arr  - inductor current series [A], shape (T,)
#   vO_arr  - output voltage series [V], shape (T,)
#   seq_len - number of time steps per input window
#   stride  - steps to advance between consecutive windows
# Returns (X, y):
#   X - float32 array (n_windows, seq_len, 3); each row is a
#       sequence of [D, iL, vO] vectors
#   y - float32 array (n_windows,); vO one step past each window
def _windows(D_arr, iL_arr, vO_arr, seq_len, stride):
    data = np.stack([D_arr, iL_arr, vO_arr], axis=1)
    T    = len(data)
    X, y = [], []
    for s in range(0, T - seq_len, stride):
        X.append(data[s : s + seq_len])
        y.append(vO_arr[s + seq_len])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32))


# Generates RNN training data by running simulate() across a range
# of operating conditions, extracting sliding windows from each
# run, and saving to a PyTorch .pt file. Covers startup transients,
# steady state at 16 (Vin, RLoad) combinations, Vref steps, load
# steps, and Vin steps to span the full operating envelope.
#   seq_len   - number of periods per input sequence; sets how
#               far back in time the RNN can look
#   stride    - steps between consecutive windows; smaller means
#               more overlap and more samples
#   save_path - file path where the .pt dataset will be written
# Returns (X, y) as PyTorch float32 tensors:
#   X - shape (N, seq_len, 3); sequences of [D, iL, vO] vectors
#   y - shape (N, 1); output voltage [V] one period after the
#       end of each sequence
def make_dataset(seq_len=50, stride=5,
                 save_path="buckConverterModel/buckConverterData.pt"):
    all_X, all_y = [], []

    # Extracts windows from one simulate() result and appends to
    # all_X and all_y. Receives the full 4-tuple from simulate()
    # but ignores vC_arr (3rd element) since _windows only needs
    # D, iL, and vO. vC_arr is used by collect_step, which passes
    # it as vC0 to the chained simulate() call.
    def collect(D_arr, iL_arr, _, vO_arr):
        Xw, yw = _windows(D_arr, iL_arr, vO_arr, seq_len, stride)
        all_X.append(Xw)
        all_y.append(yw)

    # Simulates a disturbance by chaining two back-to-back episodes
    # with different Vin and RLoad. The second episode starts from
    # the final state of the first, so the converter is settled
    # when the step occurs (matching paper Figs. 13 and 14).
    #   Vin1, Vin2     - input voltage before/after the step [V]
    #   RLoad1, RLoad2 - load resistance before/after step [Ohm]
    #   Vref           - set-point, constant across both halves [V]
    #   n              - periods in each half of the episode
    def collect_step(Vin1, Vin2, RLoad1, RLoad2, Vref, n=3000):
        r1 = simulate(Vin1, RLoad1, Vref, n)
        r2 = simulate(Vin2, RLoad2, Vref, n,
                      iL0=r1[1][-1], vC0=r1[2][-1], D0=r1[0][-1])
        collect(
            np.concatenate([r1[0], r2[0]]),
            np.concatenate([r1[1], r2[1]]),
            np.concatenate([r1[2], r2[2]]),
            np.concatenate([r1[3], r2[3]]),
        )

    # Startup + steady state across the operating envelope
    # (load range 5-50 W at 5 V -> RLoad 0.5-5 Ohm).
    for Vin in [9.0, 12.0, 15.0, 19.0]:
        for RLoad in [0.5, 1.0, 2.0, 5.0]:
            collect(*simulate(Vin, RLoad, Vref_nom, 5000))

    # Reference voltage steps.
    for Vref in [3.0, 4.0, 5.0, 6.0]:
        collect(*simulate(Vin_nom, 1.0, Vref, 5000))

    # Load steps (paper Fig. 13).
    collect_step(19.0, 19.0, 1.0, 0.5, Vref_nom)
    collect_step(19.0, 19.0, 5.0, 1.0, Vref_nom)
    collect_step(19.0, 19.0, 2.0, 0.5, Vref_nom)

    # Input voltage steps (paper Fig. 14).
    collect_step(19.0, 9.0, 1.0, 1.0, Vref_nom)
    collect_step(9.0, 19.0, 1.0, 1.0, Vref_nom)

    X = torch.from_numpy(np.concatenate(all_X))
    y = torch.from_numpy(np.concatenate(all_y)).unsqueeze(1)

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y": y}, path)

    print(f"Saved {len(X):,} sequences -> {path}")
    print(f"  X shape  : {tuple(X.shape)}")
    print(f"  y shape  : {tuple(y.shape)}")
    print(f"  vO range : {y.min().item():.4f} - {y.max().item():.4f} V")
    print(f"  iL range : {X[:,:,1].min():.4f} - {X[:,:,1].max():.4f} A")
    print(f"  D  range : {X[:,:,0].min():.4f} - {X[:,:,0].max():.4f}")
    return X, y


if __name__ == "__main__":
    plot_waveforms()
    make_dataset()
