"""
teleop_diagnostic.py — live visualisation for diagnosing where the
teleop pipeline goes wrong.

Reads from two shared memories:
    teleop_shm   — raw pika pose stream  (from pose_publisher_gui.py)
    teleop_diag  — robot state snapshot  (from mujoco_teleop_sim.py)

Shows four live matplotlib panels:

    ┌──────────────────────┬──────────────────────┐
    │  EE pos vs target    │  IK tracking error   │
    │  (3 axes over time)  │  (pos mm + rot deg)  │
    ├──────────────────────┼──────────────────────┤
    │  Joint angles        │  Pika raw input      │
    │  with limits         │  (3 axes over time)  │
    └──────────────────────┴──────────────────────┘

Plus a status bar:  engaged | tier | base pos | gripper | sample rate

Run AFTER mujoco_teleop_sim.py has started:

    python3 teleop_diagnostic.py
"""
from __future__ import annotations
import argparse
import time
from collections import deque
from multiprocessing import shared_memory, resource_tracker


# ----------------------------------------------------------------------------
# Workaround for Python bug 38119: when a process *attaches* to an existing
# shared memory segment (create=False), resource_tracker still registers it
# and warns "leaked" at shutdown — even though we correctly never unlink
# (because the owner is the publisher / sim).  Patch it to ignore shm.
# ----------------------------------------------------------------------------
def _silence_shm_resource_tracker():
    _orig_register = resource_tracker.register
    _orig_unregister = resource_tracker.unregister

    def _register(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_register(name, rtype)

    def _unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_unregister(name, rtype)

    resource_tracker.register = _register
    resource_tracker.unregister = _unregister
    if hasattr(resource_tracker, "_CLEANUP_FUNCS") and \
            "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


_silence_shm_resource_tracker()


# ----------------------------------------------------------------------------
# Force an interactive matplotlib backend BEFORE importing pyplot — otherwise
# headless servers default to 'agg' and plt.show() silently does nothing.
# ----------------------------------------------------------------------------
import os
if not os.environ.get("DISPLAY"):
    print("[diag] ⚠  DISPLAY is unset — connect with `ssh -Y` so X11 forwards.")

import matplotlib
for _bk in ("Qt5Agg", "QtAgg", "TkAgg", "GTK3Agg"):
    try:
        matplotlib.use(_bk, force=True)
        print(f"[diag] matplotlib backend: {_bk}")
        break
    except Exception as _e:
        continue
else:
    print(f"[diag] ⚠  no interactive matplotlib backend available "
          f"(current: {matplotlib.get_backend()}). Install one of: PyQt5, tk.")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# Must match mujoco_teleop_sim.py
PIKA_SHM_NAME = "teleop_shm"
PIKA_N_FLOATS = 9
DIAG_SHM_NAME = "teleop_diag"
DIAG_N_FLOATS = 32

# MJCF joint limits (must match mobile_manipulator.xml)
JOINT_NAMES  = ["j1", "j2", "j3", "j4", "j5", "j6"]
JOINT_LIMITS = [
    (-2.618, +2.618),
    ( 0.000, +3.140),
    (-2.697,  0.000),
    (-1.832, +1.832),
    (-1.220, +1.220),
    (-3.140, +3.140),
]

WINDOW_SEC = 10.0
SAMPLE_HZ  = 30.0
N_HIST     = int(WINDOW_SEC * SAMPLE_HZ)
PLOT_HZ    = 10.0


def attach(name: str, n_floats: int, retries: int = 30):
    for i in range(retries):
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            buf = np.ndarray((n_floats,), dtype=np.float32, buffer=shm.buf)
            return shm, buf
        except FileNotFoundError:
            print(f"[diag] waiting for shm '{name}'… ({i+1}/{retries})")
            time.sleep(1.0)
    raise RuntimeError(f"shm '{name}' not found after {retries}s — "
                       f"is mujoco_teleop_sim.py / pose_publisher_gui.py running?")


class History:
    """Ring buffers for time-series plots."""
    def __init__(self, n=N_HIST):
        self.n = n
        self.t      = deque(maxlen=n)
        self.tgt    = deque(maxlen=n)    # (x,y,z)
        self.ee     = deque(maxlen=n)
        self.err_p  = deque(maxlen=n)    # m
        self.err_r  = deque(maxlen=n)    # rad
        self.pika   = deque(maxlen=n)    # raw pika xyz
        self.sample_times = deque(maxlen=20)

    def push(self, t, tgt, ee, err_p, err_r, pika):
        self.t.append(t)
        self.tgt.append(tgt)
        self.ee.append(ee)
        self.err_p.append(err_p)
        self.err_r.append(err_r)
        self.pika.append(pika)
        self.sample_times.append(time.time())

    def rate(self):
        if len(self.sample_times) < 2:
            return 0.0
        dt = self.sample_times[-1] - self.sample_times[0]
        return (len(self.sample_times) - 1) / dt if dt > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=float, default=WINDOW_SEC,
                    help="seconds of history to show")
    args = p.parse_args()
    n_hist = int(args.window * SAMPLE_HZ)

    # Attach both shms
    print(f"[diag] attaching shms…")
    pika_shm, pika_buf = attach(PIKA_SHM_NAME, PIKA_N_FLOATS)
    diag_shm, diag_buf = attach(DIAG_SHM_NAME, DIAG_N_FLOATS)
    print(f"[diag] attached.")

    hist = History(n=n_hist)
    last_t = -1.0

    # --- Build figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.canvas.manager.set_window_title("Teleop diagnostic")

    ax_ee, ax_err = axes[0, 0], axes[0, 1]
    ax_jt, ax_pk  = axes[1, 0], axes[1, 1]

    colors = ['#d62728', '#2ca02c', '#1f77b4']           # x=red, y=green, z=blue
    labels = ['x', 'y', 'z']

    # --- EE pos vs target ---
    ax_ee.set_title("EE position — target (solid) vs achieved (dashed)")
    ax_ee.set_xlabel("t (s)"); ax_ee.set_ylabel("m")
    line_tgt, line_ee = [], []
    for i in range(3):
        l1, = ax_ee.plot([], [], color=colors[i], linewidth=1.6,
                          label=f'tgt_{labels[i]}')
        l2, = ax_ee.plot([], [], color=colors[i], linewidth=1.0, linestyle='--',
                          label=f'EE_{labels[i]}')
        line_tgt.append(l1); line_ee.append(l2)
    ax_ee.legend(loc='upper left', fontsize=8, ncol=2)
    ax_ee.grid(True, alpha=0.3)

    # --- Error over time ---
    ax_err.set_title("IK tracking error")
    ax_err.set_xlabel("t (s)")
    line_errp, = ax_err.plot([], [], color='#1f77b4', linewidth=1.5, label='pos (mm)')
    ax_err.set_ylabel("pos err (mm)", color='#1f77b4')
    ax_err.tick_params(axis='y', labelcolor='#1f77b4')
    ax_err2 = ax_err.twinx()
    line_errr, = ax_err2.plot([], [], color='#d62728', linewidth=1.5, label='rot (deg)')
    ax_err2.set_ylabel("rot err (deg)", color='#d62728')
    ax_err2.tick_params(axis='y', labelcolor='#d62728')
    ax_err.grid(True, alpha=0.3)
    # Threshold lines
    ax_err.axhline(50,  color='orange', linewidth=0.7, linestyle=':', alpha=0.7)
    ax_err.axhline(200, color='red',    linewidth=0.7, linestyle=':', alpha=0.7)

    # --- Joints ---
    ax_jt.set_title("Joint angles (red bands = limit zone)")
    ax_jt.set_ylabel("rad")
    ax_jt.set_ylim(-3.5, 3.5)
    ax_jt.axhline(0, color='k', linewidth=0.5)
    bars = ax_jt.bar(JOINT_NAMES, [0]*6, color='steelblue', alpha=0.85)
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        # Light grey shaded region inside limits
        ax_jt.fill_between([i-0.4, i+0.4], lo, hi, color='lightgrey', alpha=0.3, zorder=0)
        # Hard limit lines
        ax_jt.plot([i-0.4, i+0.4], [lo, lo], 'r-', linewidth=1.2, alpha=0.8)
        ax_jt.plot([i-0.4, i+0.4], [hi, hi], 'r-', linewidth=1.2, alpha=0.8)
    ax_jt.grid(True, alpha=0.3, axis='y')

    # --- Pika raw input over time ---
    ax_pk.set_title("Pika raw input (before filter)")
    ax_pk.set_xlabel("t (s)"); ax_pk.set_ylabel("m")
    line_pk = []
    for i in range(3):
        l, = ax_pk.plot([], [], color=colors[i], linewidth=1.5,
                         label=f'pika_{labels[i]}')
        line_pk.append(l)
    ax_pk.legend(loc='upper left', fontsize=8)
    ax_pk.grid(True, alpha=0.3)

    status = fig.text(
        0.5, 0.005,
        "waiting for first sample…",
        ha='center', fontsize=10, family='monospace',
    )

    def update(_frame):
        nonlocal last_t
        diag = np.array(diag_buf, copy=True)
        pika = np.array(pika_buf, copy=True)

        target_pos = diag[0:3]
        ee_pos     = diag[7:10]
        joints     = diag[14:20]
        base_x, base_y, base_yaw = diag[20], diag[21], diag[22]
        gripper_in = float(diag[23])
        tier       = int(diag[24])
        err_pos    = float(diag[25])
        err_rot    = float(diag[26])
        engaged    = diag[27] > 0.5
        t          = float(diag[28])
        pika_raw   = diag[29:32]

        # Only push when new sample arrives (sim's clock advanced)
        if engaged and t > last_t:
            last_t = t
            hist.push(t, target_pos.copy(), ee_pos.copy(),
                      err_pos, err_rot, pika_raw.copy())

        if not hist.t:
            status.set_text("waiting for engaged sim…")
            return

        ts = np.array(hist.t)
        tgts = np.array(hist.tgt)
        ees  = np.array(hist.ee)
        for i in range(3):
            line_tgt[i].set_data(ts, tgts[:, i])
            line_ee[i].set_data(ts, ees[:, i])
            line_pk[i].set_data(ts, np.array(hist.pika)[:, i])

        ax_ee.relim(); ax_ee.autoscale_view()
        ax_pk.relim(); ax_pk.autoscale_view()

        errs_mm  = np.array(hist.err_p) * 1000
        errs_deg = np.degrees(np.array(hist.err_r))
        line_errp.set_data(ts, errs_mm)
        line_errr.set_data(ts, errs_deg)
        ax_err.relim(); ax_err.autoscale_view()
        ax_err2.relim(); ax_err2.autoscale_view()

        # Joints with limit-proximity highlighting
        for i, b in enumerate(bars):
            v = float(joints[i])
            b.set_height(v)
            lo, hi = JOINT_LIMITS[i]
            span = hi - lo
            margin = 0.05 * span
            if v < lo + margin or v > hi - margin:
                b.set_color('orangered')
            else:
                b.set_color('steelblue')

        rate = hist.rate()
        status.set_text(
            f"engaged={engaged}  tier={tier}  "
            f"base=(x={base_x:+.2f}m, y={base_y:+.2f}m, yaw={np.degrees(base_yaw):+.1f}°)  "
            f"grip_in={gripper_in:.2f}  "
            f"err_pos={err_pos*1000:.1f}mm  err_rot={np.degrees(err_rot):.1f}°  "
            f"shm_rate={rate:.1f}Hz"
        )

    ani = FuncAnimation(fig, update, interval=int(1000 / PLOT_HZ),
                         blit=False, cache_frame_data=False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        pika_shm.close()
        diag_shm.close()


if __name__ == "__main__":
    main()
