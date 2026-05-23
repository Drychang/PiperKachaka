"""
teleop_ghost_viewer.py — "digital twin" MuJoCo viewer for the bridge.

NO IK is solved here.  This process just:
  1. reads the diag shared memory (published by mujoco_teleop_sim.py or by
     ros2_bridge.py on the real robot)
  2. writes the published joint angles + base pose straight into MuJoCo data
  3. moves the target mocap to the published 6D target pose
  4. renders

Use case: while the real piper + kachaka are running, you keep this window
open as a 3D mirror of what the bridge thinks the robot is doing —
target ball + commanded joint positions side-by-side with the real arm.

Run:
    python3 teleop_ghost_viewer.py

Diag shm layout (32 × float32, same as written by mujoco_teleop_sim.py):
    [0:3]   target_pos       (world)
    [3:7]   target_quat_xyzw
    [7:10]  EE_pos           (current, FK)
    [10:14] EE_quat_xyzw
    [14:20] joint1..joint6   (rad, commanded)
    [20:23] base_x, base_y, base_yaw
    [23]    gripper_in (pika unit 0..0.986)
    [24]    last_tier
    [25]    pos_err (m)
    [26]    rot_err (rad)
    [27]    engaged flag
    [28]    timestamp (s)
    [29:32] pika_raw_xyz
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from multiprocessing import shared_memory, resource_tracker
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


CSV_HEADER = [
    "t_sec",
    "target_x", "target_y", "target_z",
    "target_qx", "target_qy", "target_qz", "target_qw",
    "ee_x", "ee_y", "ee_z",
    "ee_qx", "ee_qy", "ee_qz", "ee_qw",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    "base_x", "base_y", "base_yaw",
    "gripper_in",
    "tier",
    "err_pos_m", "err_rot_rad",
    "engaged",
    "pika_raw_x", "pika_raw_y", "pika_raw_z",
]


def open_csv_log(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(log_dir) / f"teleop_ghost_{ts}.csv"
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(CSV_HEADER)
    return f, w, path


DIAG_SHM_NAME = "teleop_diag"
DIAG_N_FLOATS = 32
RENDER_HZ     = 30.0

# Pika gripper range mapping to MJCF joint7 range (matches SimArmIO)
GRIPPER_IN_OPEN  = 0.986
GRIPPER_OUT_OPEN = 0.035


def _silence_shm_resource_tracker():
    _orig_register   = resource_tracker.register
    _orig_unregister = resource_tracker.unregister
    def _register(name, rtype):
        if rtype == "shared_memory": return
        return _orig_register(name, rtype)
    def _unregister(name, rtype):
        if rtype == "shared_memory": return
        return _orig_unregister(name, rtype)
    resource_tracker.register   = _register
    resource_tracker.unregister = _unregister
    if hasattr(resource_tracker, "_CLEANUP_FUNCS") and \
            "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]
_silence_shm_resource_tracker()


def attach_diag_shm(retries: int = 30):
    for i in range(retries):
        try:
            return shared_memory.SharedMemory(name=DIAG_SHM_NAME, create=False)
        except FileNotFoundError:
            print(f"[ghost] waiting for diag shm '{DIAG_SHM_NAME}'… "
                  f"({i+1}/{retries}) — start mujoco_teleop_sim.py or ros2_bridge.py")
            time.sleep(1.0)
    raise RuntimeError(f"diag shm '{DIAG_SHM_NAME}' not found.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                    help="MJCF (defaults to mobile_manipulator.xml next to this file)")
    p.add_argument("--log-dir", default="logs",
                    help="folder for per-run CSV logs (relative to script dir)")
    p.add_argument("--no-log", action="store_true",
                    help="disable CSV logging")
    args = p.parse_args()

    here = Path(__file__).parent
    model_path = args.model or str(here / "mobile_manipulator.xml")
    if not Path(model_path).exists():
        sys.exit(f"FATAL: model {model_path!r} not found")

    shm = attach_diag_shm()
    buf = np.ndarray((DIAG_N_FLOATS,), dtype=np.float32, buffer=shm.buf)
    print(f"[ghost] attached to '{DIAG_SHM_NAME}'")

    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    # Joint qpos addresses (base + arm)
    joint_names = ["base_x", "base_y", "base_yaw",
                    "joint1", "joint2", "joint3",
                    "joint4", "joint5", "joint6"]
    qpos_addrs = []
    for n in joint_names:
        jid = model.joint(n).id
        qpos_addrs.append(int(model.jnt_qposadr[jid]))

    # Gripper joint
    j7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
    gripper_qpos_addr = int(model.jnt_qposadr[j7_id])

    # Target mocap
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap = int(model.body_mocapid[target_body])

    # EE site (for distance overlay)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")

    # ---- CSV logger ----
    csv_file = None; csv_writer = None; csv_path = None
    csv_rows_since_flush = 0
    last_t_logged = -1.0
    if not args.no_log:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = here / log_dir
        csv_file, csv_writer, csv_path = open_csv_log(str(log_dir))

    print("=" * 60)
    print(" Teleop ghost viewer")
    print("=" * 60)
    print(f"  model:        {model_path}")
    print(f"  diag shm:     {DIAG_SHM_NAME}")
    print(f"  render rate:  {RENDER_HZ:.0f} Hz")
    print(f"  csv log:      {csv_path if csv_path else '(disabled)'}")
    print("  no IK is solved — purely visualises shm content")
    print("=" * 60)

    dt = 1.0 / RENDER_HZ
    last_log = 0.0

    with mujoco.viewer.launch_passive(
        model=model, data=data,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        try:
            while viewer.is_running():
                t0 = time.time()
                d = np.array(buf, copy=True)

                target_pos       = d[0:3].astype(np.float64)
                target_quat_xyzw = d[3:7].astype(np.float64)
                joints           = d[14:20].astype(np.float64)
                base_x, base_y, base_yaw = float(d[20]), float(d[21]), float(d[22])
                gripper_in       = float(d[23])
                tier             = int(d[24])
                err_pos          = float(d[25])
                err_rot          = float(d[26])
                engaged          = d[27] > 0.5

                # Skip first frames before bridge engages (avoid quat=0)
                qn = float(np.linalg.norm(target_quat_xyzw))
                if not engaged or qn < 1e-6:
                    viewer.sync()
                    time.sleep(dt)
                    continue
                target_quat_xyzw /= qn

                # ---- Apply commanded state to mujoco data ----
                data.qpos[qpos_addrs[0]] = base_x
                data.qpos[qpos_addrs[1]] = base_y
                data.qpos[qpos_addrs[2]] = base_yaw
                for i in range(6):
                    data.qpos[qpos_addrs[3 + i]] = joints[i]

                # Map pika gripper [0, 0.986] → mjcf joint7 [0, 0.035]
                grip_sim = max(0.0, min(GRIPPER_OUT_OPEN,
                                          gripper_in / GRIPPER_IN_OPEN * GRIPPER_OUT_OPEN))
                data.qpos[gripper_qpos_addr] = grip_sim

                # ---- Apply target mocap (xyzw → wxyz for MuJoCo) ----
                data.mocap_pos[target_mocap]  = target_pos
                data.mocap_quat[target_mocap] = np.array([
                    target_quat_xyzw[3], target_quat_xyzw[0],
                    target_quat_xyzw[1], target_quat_xyzw[2]])

                mujoco.mj_forward(model, data)
                viewer.sync()

                # ---- Read EE pose from mujoco FK + read shm timestamp ----
                cur_ee = data.site_xpos[ee_site_id].copy()
                cur_ee_mat = data.site_xmat[ee_site_id].reshape(3, 3).copy()
                # convert ee_mat to quat (xyzw via scipy-free path)
                # We can read sim's published EE quat directly from diag for consistency
                ee_quat_xyzw_pub = d[10:14].astype(np.float64)
                ee_n = float(np.linalg.norm(ee_quat_xyzw_pub))
                if ee_n > 1e-6:
                    ee_quat_xyzw_pub /= ee_n

                pika_raw = d[29:32].astype(np.float64)
                t_shm    = float(d[28])
                pika_raw_x, pika_raw_y, pika_raw_z = pika_raw[0], pika_raw[1], pika_raw[2]

                # ---- CSV row (only when shm timestamp advances) ----
                if csv_writer is not None and t_shm > last_t_logged:
                    last_t_logged = t_shm
                    csv_writer.writerow([
                        f"{t_shm:.4f}",
                        f"{target_pos[0]:.6f}", f"{target_pos[1]:.6f}", f"{target_pos[2]:.6f}",
                        f"{target_quat_xyzw[0]:.6f}", f"{target_quat_xyzw[1]:.6f}",
                        f"{target_quat_xyzw[2]:.6f}", f"{target_quat_xyzw[3]:.6f}",
                        f"{cur_ee[0]:.6f}", f"{cur_ee[1]:.6f}", f"{cur_ee[2]:.6f}",
                        f"{ee_quat_xyzw_pub[0]:.6f}", f"{ee_quat_xyzw_pub[1]:.6f}",
                        f"{ee_quat_xyzw_pub[2]:.6f}", f"{ee_quat_xyzw_pub[3]:.6f}",
                        f"{joints[0]:.6f}", f"{joints[1]:.6f}", f"{joints[2]:.6f}",
                        f"{joints[3]:.6f}", f"{joints[4]:.6f}", f"{joints[5]:.6f}",
                        f"{base_x:.6f}", f"{base_y:.6f}", f"{base_yaw:.6f}",
                        f"{gripper_in:.4f}",
                        tier,
                        f"{err_pos:.6f}", f"{err_rot:.6f}",
                        int(bool(engaged)),
                        f"{pika_raw_x:.6f}", f"{pika_raw_y:.6f}", f"{pika_raw_z:.6f}",
                    ])
                    csv_rows_since_flush += 1
                    if csv_rows_since_flush >= 30:           # ~1 s
                        csv_file.flush()
                        csv_rows_since_flush = 0

                # Periodic console summary
                now = time.time()
                if now - last_log > 1.0:
                    last_log = now
                    print(f"[ghost] tgt=({target_pos[0]:+.2f},{target_pos[1]:+.2f},"
                          f"{target_pos[2]:+.2f}) "
                          f"EE=({cur_ee[0]:+.2f},{cur_ee[1]:+.2f},{cur_ee[2]:+.2f}) "
                          f"err_pos={err_pos*1000:.1f}mm err_rot={np.degrees(err_rot):.1f}° "
                          f"tier={tier}  base=({base_x:+.2f},{base_y:+.2f},"
                          f"{np.degrees(base_yaw):+.0f}°)")

                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
        except KeyboardInterrupt:
            pass
        finally:
            print("\n[ghost] exiting.")
            try:
                shm.close()
            except Exception:
                pass
            if csv_file is not None:
                try:
                    csv_file.flush()
                    csv_file.close()
                    print(f"[ghost] csv saved: {csv_path}")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
