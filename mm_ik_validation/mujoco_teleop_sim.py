"""
mujoco_teleop_sim.py — robot-side MuJoCo sim.

Reads the 6D pose stream that pose_publisher_gui.py writes to shared
memory and tracks it with the kachaka + piper whole-body IK, mimicking
exactly what ros2_bridge.py does on the real robot:

  1. read raw pose from shared memory (replaces /pika/pose)
  2. windowed-mean smoothing on pos + SVD mean on rotation
       (from the teleopt_tool.py reference design)
  3. per-frame incremental delta:  Δ = pika_curr − pika_last
  4. integrate Δ onto an internal target EE pose (no fixed origin)
  5. run mink whole-body IK with heavy base damping
       (FORCE_ARM_ONLY → base never moves)
  6. step MuJoCo data so the viewer shows the robot following

Run (with pose_publisher_gui.py already running in another shell):

    python3 mujoco_teleop_sim.py

Options:
    --model PATH                 (default: mobile_manipulator.xml)
    --force-arm-only / --allow-base
    --ori-cost FLOAT             (default 0.1; lower = position-priority)
    --pos-scale FLOAT            (default 1.0)
    --filter-window N            (default 3)
"""
from __future__ import annotations
import argparse
import sys
import time
from multiprocessing import shared_memory, resource_tracker
from pathlib import Path


# Workaround for Python bug 38119: attaching to an existing shm (create=False)
# registers it with resource_tracker, which then warns "leaked" at exit even
# though the owner properly unlinks.  Patch out shm tracking.
def _silence_shm_resource_tracker():
    _orig_register   = resource_tracker.register
    _orig_unregister = resource_tracker.unregister
    def _register(name, rtype):
        if rtype == "shared_memory": return
        return _orig_register(name, rtype)
    def _unregister(name, rtype):
        if rtype == "shared_memory": return
        return _orig_unregister(name, rtype)
    resource_tracker.register = _register
    resource_tracker.unregister = _unregister
    if hasattr(resource_tracker, "_CLEANUP_FUNCS") and \
            "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]
_silence_shm_resource_tracker()

import mink
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from whole_body_ik import WholeBodyIK                       # noqa: E402
from whole_body_executor import (                              # noqa: E402
    SimBaseIO, SimArmIO, StreamingController,
)


SHM_NAME            = "teleop_shm"
N_FLOATS            = 9
DIAG_SHM_NAME       = "teleop_diag"
DIAG_N_FLOATS       = 32         # see below for layout
CONTROL_RATE_HZ     = 30.0
PIKA_POSITION_SCALE = 1.0
PIKA_FILTER_WINDOW  = 3
PIKA_PER_FRAME_MAX  = 0.20     # m, frame-to-frame reject (6m/s @ 30Hz)
LOCKED_ARM_JOINTS   = ["joint5"]
TIER1_BASE_DAMPING  = 500.0
TIER2_BASE_DAMPING  = 5.0
FALLBACK_POS_THRESH = 0.15
FALLBACK_ROT_THRESH = 0.30
STREAM_MAX_V        = 0.10
STREAM_MAX_W        = 0.30
STREAM_MAX_ARM_VEL  = 1.0

# Gripper rescaling
GRIPPER_IN_CLOSED  = 0.0
GRIPPER_IN_OPEN    = 0.986
GRIPPER_OUT_CLOSED = 0.0
GRIPPER_OUT_OPEN   = 0.035   # SimArmIO MJCF range


# ---------------------------------------------------------------------------
# Pose smoothing — windowed mean for position, SVD mean for rotation.
# (Direct port of PoseFilterWindow from teleopt_tool.py reference.)
# ---------------------------------------------------------------------------
class PoseFilterWindow:
    def __init__(self, window: int = 5):
        self.window = max(1, int(window))
        self.pos_buf: list[np.ndarray] = []
        self.quat_buf: list[np.ndarray] = []

    def filter(self, pos: np.ndarray, quat_xyzw: np.ndarray):
        self.pos_buf.append(pos)
        if len(self.pos_buf) > self.window:
            self.pos_buf.pop(0)
        pos_avg = np.mean(self.pos_buf, axis=0)

        self.quat_buf.append(quat_xyzw)
        if len(self.quat_buf) > self.window:
            self.quat_buf.pop(0)
        mats = [R.from_quat(q).as_matrix() for q in self.quat_buf]
        mat_avg = np.mean(mats, axis=0)
        u, _, vh = np.linalg.svd(mat_avg)
        rot = u @ vh
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1
            rot = u @ vh
        return pos_avg, R.from_matrix(rot).as_quat()

    def reset(self):
        self.pos_buf.clear()
        self.quat_buf.clear()


def attach_shm(retries: int = 30):
    for i in range(retries):
        try:
            return shared_memory.SharedMemory(name=SHM_NAME, create=False)
        except FileNotFoundError:
            print(f"[sim] Waiting for shm '{SHM_NAME}'… "
                  f"({i+1}/{retries}) — start pose_publisher_gui.py")
            time.sleep(1.0)
    raise RuntimeError(f"Could not attach to shared memory '{SHM_NAME}'.")


def map_gripper(x: float) -> float:
    span_in = GRIPPER_IN_OPEN - GRIPPER_IN_CLOSED
    if abs(span_in) < 1e-9:
        return float(np.clip(x, GRIPPER_OUT_CLOSED, GRIPPER_OUT_OPEN))
    ratio = (x - GRIPPER_IN_CLOSED) / span_in
    ratio = max(0.0, min(1.0, ratio))
    return GRIPPER_OUT_CLOSED + ratio * (GRIPPER_OUT_OPEN - GRIPPER_OUT_CLOSED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobile_manipulator.xml")
    p.add_argument("--force-arm-only", dest="force_arm_only",
                    action="store_true", default=True)
    p.add_argument("--allow-base", dest="force_arm_only",
                    action="store_false")
    p.add_argument("--ori-cost", type=float, default=0.1)
    p.add_argument("--pos-scale", type=float, default=PIKA_POSITION_SCALE)
    p.add_argument("--filter-window", type=int, default=PIKA_FILTER_WINDOW)
    p.add_argument("--remap-mode", choices=["none", "auto", "explicit"],
                    default="none",
                    help="pika-frame → robot-frame remap. "
                         "'none'=raw world delta (reference tool style); "
                         "'auto'=use pika orientation @ engage to align with robot; "
                         "'explicit'=use --remap-rpy below.")
    p.add_argument("--remap-rpy", type=float, nargs=3, metavar=("R", "P", "Y"),
                    default=[0.0, 0.0, 0.0],
                    help="(deg) explicit Euler angles for R_pika_to_robot.")
    p.add_argument("--locked-joints", type=str, default="joint5",
                    help="comma-sep joint names to lock (e.g. 'joint5' or '' for none)")
    args = p.parse_args()
    # Convert "" / "joint5,joint3" → list
    locked_arm_joints = [s.strip() for s in args.locked_joints.split(",") if s.strip()]

    # Build the remap rotation
    if args.remap_mode == "explicit":
        R_pika_to_robot = R.from_euler("xyz", args.remap_rpy, degrees=True)
    else:
        R_pika_to_robot = R.identity()        # 'none' or initial 'auto' — finalised at engage

    shm = attach_shm()
    buf = np.ndarray((N_FLOATS,), dtype=np.float32, buffer=shm.buf)

    # Diagnostic shm — published for teleop_diagnostic.py.  Layout (32 × float32):
    #   [0:3]   target_pos           (world)
    #   [3:7]   target_quat_xyzw
    #   [7:10]  EE_pos               (world, current achieved)
    #   [10:14] EE_quat_xyzw
    #   [14:20] joint1..joint6       (rad)
    #   [20:23] base x, y, yaw       (m, m, rad)
    #   [23]    gripper_in (pika unit)
    #   [24]    last_tier (1 or 2)
    #   [25]    pos_err (m)
    #   [26]    rot_err (rad)
    #   [27]    engaged flag (0/1)
    #   [28]    monotonic timestamp (s)
    #   [29]    pika_raw_x          (last raw pika input before filter)
    #   [30]    pika_raw_y
    #   [31]    pika_raw_z
    try:
        diag_shm = shared_memory.SharedMemory(
            name=DIAG_SHM_NAME, create=True, size=DIAG_N_FLOATS * 4)
        diag_owned = True
    except FileExistsError:
        diag_shm = shared_memory.SharedMemory(name=DIAG_SHM_NAME, create=False)
        diag_owned = False
    diag_buf = np.ndarray((DIAG_N_FLOATS,), dtype=np.float32, buffer=diag_shm.buf)
    diag_buf[:] = 0
    print(f"[sim] diag shm '{DIAG_SHM_NAME}' "
          f"{'created' if diag_owned else 'attached'}, "
          f"size={DIAG_N_FLOATS * 4} bytes")
    t_start = time.time()

    # Build IK and sim
    ik = WholeBodyIK(args.model, pos_cost=1.0, ori_cost=args.ori_cost)
    base_io = SimBaseIO(ik)
    arm_io  = SimArmIO(ik)
    streamer = StreamingController(
        ik, base_io, arm_io,
        max_v=STREAM_MAX_V, max_w=STREAM_MAX_W,
        yaw_assist_gain=0.5, inner_iters=5,
        max_arm_vel=STREAM_MAX_ARM_VEL,
        tier1_base_damping=TIER1_BASE_DAMPING,
        tier2_base_damping=TIER2_BASE_DAMPING,
        fallback_pos_threshold=FALLBACK_POS_THRESH,
        fallback_rot_threshold=FALLBACK_ROT_THRESH,
        force_arm_only=args.force_arm_only,
        locked_arm_joints=locked_arm_joints,
    )

    # State (mirrors what ros2_bridge.py keeps internally)
    engaged          = False
    pose_filter      = PoseFilterWindow(window=args.filter_window)
    target_pos       = None       # (3,)  mutable
    target_rot       = None       # scipy Rotation
    pika_last_pos    = None
    pika_last_rot    = None
    last_reset_count = 0.0
    last_warned      = False

    dt        = 1.0 / CONTROL_RATE_HZ
    last_dbg  = 0.0

    print("=" * 60)
    print(" MuJoCo whole-body teleop sim")
    print("=" * 60)
    print(f" shm:               {SHM_NAME}")
    print(f" model:             {args.model}")
    print(f" FORCE_ARM_ONLY:    {args.force_arm_only}")
    print(f" locked joints:     {locked_arm_joints}")
    print(f" ori_cost / pos:    {args.ori_cost} / 1.0")
    print(f" pos scale:         {args.pos_scale}")
    print(f" filter window:     {args.filter_window} samples")
    print(f" remap mode:        {args.remap_mode}"
          + (f"  (rpy={args.remap_rpy}°)" if args.remap_mode == 'explicit' else ''))
    print("=" * 60)

    with mujoco.viewer.launch_passive(
        model=ik.model, data=ik.data,
        show_left_ui=True, show_right_ui=False,
    ) as viewer:
        # Park the target mocap on the EE so engage doesn't visually jump
        mink.move_mocap_to_frame(ik.model, ik.data, "target", "ee", "site")
        mujoco.mj_forward(ik.model, ik.data)

        try:
            while viewer.is_running():
                t0 = time.time()

                # ---- Read shm (single tear-resistant copy) ----
                snapshot = np.array(buf, copy=True)
                raw_pos      = snapshot[0:3].astype(np.float64)
                raw_quat_xyzw = snapshot[3:7].astype(np.float64)
                gripper_in   = float(snapshot[7])
                reset_count  = float(snapshot[8])

                # ---- Detect re-grip ----
                if reset_count != last_reset_count:
                    last_reset_count = reset_count
                    pika_last_pos = None
                    pika_last_rot = None
                    pose_filter.reset()
                    print(f"\n[sim] Re-grip received (#{int(reset_count)}) — "
                          "freezing target, re-seeding baseline.")

                # ---- Smooth ----
                # quaternion must be unit-norm before from_quat — guard against
                # all-zero shm before the publisher starts writing
                qn = np.linalg.norm(raw_quat_xyzw)
                if qn < 1e-6:
                    time.sleep(dt)
                    continue
                raw_quat_xyzw = raw_quat_xyzw / qn
                pos_f, quat_f = pose_filter.filter(raw_pos, raw_quat_xyzw)
                pika_rot = R.from_quat(quat_f)

                if not engaged:
                    # Snapshot robot EE as starting target
                    mujoco.mj_forward(ik.model, ik.data)
                    target_pos = ik.data.site_xpos[ik.ee_site_id].copy()
                    mat = ik.data.site_xmat[ik.ee_site_id].reshape(3, 3).copy()
                    target_rot = R.from_matrix(mat)
                    pika_last_pos = pos_f.copy()
                    pika_last_rot = pika_rot
                    ik.posture_task.set_target_from_configuration(ik.configuration)

                    # Auto-calibration: align pika engage orientation with robot
                    # engage orientation. From then on, "pika +x" feels like "robot +x".
                    if args.remap_mode == "auto":
                        R_pika_to_robot = target_rot * pika_rot.inv()
                        fwd = R_pika_to_robot.apply([1.0, 0.0, 0.0])
                        print(f"[sim] auto remap: pika +x → "
                              f"({fwd[0]:+.2f},{fwd[1]:+.2f},{fwd[2]:+.2f}) in robot world")

                    engaged = True
                    rpy = target_rot.as_euler("xyz", degrees=True)
                    print(f"[sim] ENGAGED.")
                    print(f"      pika baseline pos=({pos_f[0]:+.3f},"
                          f"{pos_f[1]:+.3f},{pos_f[2]:+.3f})")
                    print(f"      robot_EE      pos=({target_pos[0]:+.3f},"
                          f"{target_pos[1]:+.3f},{target_pos[2]:+.3f})  "
                          f"rpy=({rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f})°")
                elif pika_last_pos is None:
                    # Just after re-grip: this sample is the new baseline only
                    pika_last_pos = pos_f.copy()
                    pika_last_rot = pika_rot
                else:
                    pos_delta_pika = (pos_f - pika_last_pos) * args.pos_scale
                    if np.linalg.norm(pos_delta_pika) > PIKA_PER_FRAME_MAX:
                        if not last_warned:
                            print(f"\n[sim] Per-frame Δ={np.linalg.norm(pos_delta_pika):.2f}m "
                                  f"> {PIKA_PER_FRAME_MAX}m → rejecting, re-seeding.")
                            last_warned = True
                        pika_last_pos = pos_f.copy()
                        pika_last_rot = pika_rot
                    else:
                        last_warned = False
                        # Apply pika→robot frame remap to BOTH translation and rotation deltas
                        pos_delta = R_pika_to_robot.apply(pos_delta_pika)
                        rot_delta_pika = pika_rot * pika_last_rot.inv()
                        rot_delta = R_pika_to_robot * rot_delta_pika * R_pika_to_robot.inv()
                        target_pos = target_pos + pos_delta
                        target_rot = rot_delta * target_rot
                        pika_last_pos = pos_f.copy()
                        pika_last_rot = pika_rot

                # ---- Hand to streamer (xyzw → wxyz for mink) ----
                tq = target_rot.as_quat()
                quat_wxyz = np.array([tq[3], tq[0], tq[1], tq[2]])
                streamer.set_target(target_pos, quat_wxyz)
                ik.set_target_mocap(target_pos, quat_wxyz)
                streamer.set_gripper(map_gripper(gripper_in))

                # ---- IK tick (mutates ik.data.qpos) ----
                streamer.tick(dt)

                # ---- Publish diagnostic snapshot to shm ----
                cur_ee_pos, cur_ee_mat = ik.get_ee_pose()
                cur_ee_rot = R.from_matrix(cur_ee_mat)
                cur_quat_xyzw = cur_ee_rot.as_quat()
                sol = ik.get_solution()
                err_pos_m = float(np.linalg.norm(target_pos - cur_ee_pos))
                err_rot_rad = float((target_rot * cur_ee_rot.inv()).magnitude())

                diag_buf[0:3]   = target_pos.astype(np.float32)
                tq_xyzw         = target_rot.as_quat().astype(np.float32)
                diag_buf[3:7]   = tq_xyzw
                diag_buf[7:10]  = cur_ee_pos.astype(np.float32)
                diag_buf[10:14] = cur_quat_xyzw.astype(np.float32)
                diag_buf[14:20] = np.array(
                    [sol[f"joint{i}"] for i in range(1, 7)], dtype=np.float32)
                diag_buf[20]    = float(sol["base_x"])
                diag_buf[21]    = float(sol["base_y"])
                diag_buf[22]    = float(sol["base_yaw"])
                diag_buf[23]    = float(gripper_in)
                diag_buf[24]    = float(streamer.last_tier_used)
                diag_buf[25]    = err_pos_m
                diag_buf[26]    = err_rot_rad
                diag_buf[27]    = 1.0 if engaged else 0.0
                diag_buf[28]    = float(time.time() - t_start)
                diag_buf[29:32] = raw_pos.astype(np.float32)

                viewer.sync()

                # ---- 1 Hz status ----
                now = time.time()
                if now - last_dbg > 1.0:
                    last_dbg = now
                    cur_ee, _ = ik.get_ee_pose()
                    err_mm = float(np.linalg.norm(target_pos - cur_ee)) * 1000
                    sol = ik.get_solution()
                    arm = [sol[f"joint{i}"] for i in range(1, 7)]
                    print(f"[IK] tgt=({target_pos[0]:+.2f},{target_pos[1]:+.2f},"
                          f"{target_pos[2]:+.2f}) "
                          f"EE=({cur_ee[0]:+.2f},{cur_ee[1]:+.2f},{cur_ee[2]:+.2f}) "
                          f"err={err_mm:5.1f}mm "
                          f"arm=[{arm[0]:+.2f},{arm[1]:+.2f},{arm[2]:+.2f},"
                          f"{arm[3]:+.2f},{arm[4]:+.2f},{arm[5]:+.2f}] "
                          f"base=({sol['base_x']:+.2f},{sol['base_y']:+.2f},"
                          f"{np.degrees(sol['base_yaw']):+.1f}°) "
                          f"tier={streamer.last_tier_used}")

                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
        except KeyboardInterrupt:
            pass
        finally:
            print("\n[sim] exiting.")
            try:
                shm.close()
            except Exception:
                pass
            try:
                diag_shm.close()
                if diag_owned:
                    diag_shm.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
