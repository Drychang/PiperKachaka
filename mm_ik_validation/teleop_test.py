"""
teleop_test.py — headless tests for the pose-stream-→-IK pipeline.

Reproduces the same logic used by mujoco_teleop_sim.py (smoothing + per-frame
incremental delta + mink whole-body IK with FORCE_ARM_ONLY) in a no-GUI
loop, so we can stress-test the bridge against real-world failure modes:

    1. baseline           clean stream, identity frame, no noise
    2. heavy noise        2 cm Gaussian jitter — filter must keep EE smooth
    3. frame mismatch     pika frame yawed 90° vs robot — R_remap must compensate
    4. scaling mismatch   pika reports 0.1× true motion — pos_scale must compensate
    5. starting offset    pika starts at (1.5, 0.3, 0.5) — engage absorbs offset
    6. workspace edge     pika hits boundary; re-grip continues seamlessly
    7. combined           all the above at once

Run:
    cd .../mm_ik_validation
    python3 teleop_test.py
"""
from __future__ import annotations
import os
import sys
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

import mujoco
import mink

sys.path.insert(0, str(Path(__file__).parent))
from whole_body_ik import WholeBodyIK                       # noqa: E402
from whole_body_executor import SimBaseIO, SimArmIO, StreamingController  # noqa: E402


# ----------------------------------------------------------------------------
# Pose smoothing — windowed mean for position, SVD mean for rotation.
# ----------------------------------------------------------------------------
class PoseFilterWindow:
    def __init__(self, window: int = 5):
        self.window = max(1, int(window))
        self.pos_buf: list[np.ndarray] = []
        self.quat_buf: list[np.ndarray] = []

    def filter(self, pos, quat_xyzw):
        self.pos_buf.append(np.asarray(pos, dtype=np.float64))
        if len(self.pos_buf) > self.window:
            self.pos_buf.pop(0)
        pos_avg = np.mean(self.pos_buf, axis=0)

        self.quat_buf.append(np.asarray(quat_xyzw, dtype=np.float64))
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


# ----------------------------------------------------------------------------
# TeleopBridge — pure-logic version of mujoco_teleop_sim's pipeline.
# Adds an explicit R_pika_to_robot remap so we can stress-test frame mismatch.
# ----------------------------------------------------------------------------
class TeleopBridge:
    def __init__(self, model_path: str,
                 ori_cost: float = 0.1,
                 pos_scale: float = 1.0,
                 filter_window: int = 3,
                 force_arm_only: bool = True,
                 R_pika_to_robot: R | None = None,
                 locked_arm_joints=("joint5",),
                 per_frame_max: float = 0.20):
        self.ik = WholeBodyIK(model_path, ori_cost=ori_cost)
        self.base_io = SimBaseIO(self.ik)
        self.arm_io = SimArmIO(self.ik)
        self.streamer = StreamingController(
            self.ik, self.base_io, self.arm_io,
            max_v=0.10, max_w=0.30, yaw_assist_gain=0.5, inner_iters=5,
            max_arm_vel=1.0,
            tier1_base_damping=500.0, tier2_base_damping=5.0,
            fallback_pos_threshold=0.15, fallback_rot_threshold=0.30,
            force_arm_only=force_arm_only,
            locked_arm_joints=list(locked_arm_joints),
        )
        self.pose_filter = PoseFilterWindow(window=filter_window)
        self.pos_scale = pos_scale
        self.R_remap = R.identity() if R_pika_to_robot is None else R_pika_to_robot
        self.per_frame_max = per_frame_max

        self.engaged = False
        self.target_pos = None
        self.target_rot = None
        self.pika_last_pos = None
        self.pika_last_rot = None
        self.robot_initial_ee = None      # for tests to read
        self.robot_initial_rot = None

    def reset_baseline(self):
        self.pose_filter.reset()
        self.pika_last_pos = None
        self.pika_last_rot = None

    def step(self, pika_pos, pika_quat_xyzw, gripper, dt):
        q = np.asarray(pika_quat_xyzw, dtype=np.float64)
        q = q / (np.linalg.norm(q) + 1e-12)
        pos_f, quat_f = self.pose_filter.filter(pika_pos, q)
        pika_rot = R.from_quat(quat_f)

        if not self.engaged:
            mujoco.mj_forward(self.ik.model, self.ik.data)
            self.target_pos = self.ik.data.site_xpos[self.ik.ee_site_id].copy()
            mat = self.ik.data.site_xmat[self.ik.ee_site_id].reshape(3, 3).copy()
            self.target_rot = R.from_matrix(mat)
            self.robot_initial_ee = self.target_pos.copy()
            self.robot_initial_rot = R.from_matrix(mat)
            self.pika_last_pos = pos_f.copy()
            self.pika_last_rot = pika_rot
            self.ik.posture_task.set_target_from_configuration(self.ik.configuration)
            self.engaged = True
        elif self.pika_last_pos is None:
            self.pika_last_pos = pos_f.copy()
            self.pika_last_rot = pika_rot
        else:
            pos_delta_pika = (pos_f - self.pika_last_pos) * self.pos_scale
            if np.linalg.norm(pos_delta_pika) > self.per_frame_max:
                # safety reject — re-seed
                self.pika_last_pos = pos_f.copy()
                self.pika_last_rot = pika_rot
            else:
                # Apply frame remap (pika → robot)
                pos_delta_robot = self.R_remap.apply(pos_delta_pika)
                rot_delta_pika = pika_rot * self.pika_last_rot.inv()
                # Conjugate to express the rotation in robot frame
                rot_delta_robot = self.R_remap * rot_delta_pika * self.R_remap.inv()

                self.target_pos = self.target_pos + pos_delta_robot
                self.target_rot = rot_delta_robot * self.target_rot
                self.pika_last_pos = pos_f.copy()
                self.pika_last_rot = pika_rot

        tq = self.target_rot.as_quat()
        quat_wxyz = np.array([tq[3], tq[0], tq[1], tq[2]])
        self.streamer.set_target(self.target_pos, quat_wxyz)
        self.streamer.set_gripper(gripper)
        self.streamer.tick(dt)

        cur_ee, _ = self.ik.get_ee_pose()
        return cur_ee.copy(), self.target_pos.copy()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
DT = 0.033
QUAT_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def run_trajectory(bridge, pika_xyz_traj, quat_traj=None, gripper=0.0):
    """Step bridge through a given pika trajectory; return (ee, target) traces."""
    n = len(pika_xyz_traj)
    if quat_traj is None:
        quat_traj = [QUAT_IDENTITY] * n
    ee_list, tgt_list = [], []
    for i in range(n):
        ee, tgt = bridge.step(pika_xyz_traj[i], quat_traj[i], gripper, DT)
        ee_list.append(ee.copy())
        tgt_list.append(tgt.copy())
    return np.array(ee_list), np.array(tgt_list)


def smoothness(traj):
    """Mean magnitude of 3-point second difference (proxy for jerk)."""
    if len(traj) < 3:
        return 0.0
    d2 = traj[2:] - 2 * traj[1:-1] + traj[:-2]
    return float(np.linalg.norm(d2, axis=1).mean())


def make_line(start, end, n):
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    return np.array([start + (end - start) * (i / (n - 1)) for i in range(n)])


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


def t1_baseline(model):
    """Clean stream, identity frame. EE should converge to target within 50mm."""
    bridge = TeleopBridge(model)
    traj = make_line([0, 0, 0], [0.30, 0, 0], 200)   # 30cm forward over ~6s
    ee, tgt = run_trajectory(bridge, traj)
    final_err = np.linalg.norm(ee[-1] - tgt[-1])
    target_moved = np.linalg.norm(tgt[-1] - tgt[0])
    ee_moved = np.linalg.norm(ee[-1] - ee[0])
    detail = (f"target moved {target_moved*1000:.0f}mm  "
              f"EE moved {ee_moved*1000:.0f}mm  "
              f"final err {final_err*1000:.1f}mm")
    # Target moved 30cm.  EE may not reach if unreachable, but err should be < 5cm
    return TestResult("baseline", final_err < 0.05, detail)


def t2_heavy_noise(model):
    """2cm sigma Gaussian noise. Filter must keep EE smoother than no-filter."""
    rng = np.random.default_rng(42)
    base = make_line([0, 0, 0], [0.30, 0, 0], 200)
    noisy = base + rng.normal(0, 0.02, base.shape)        # 2cm jitter

    b_no  = TeleopBridge(model, filter_window=1)         # no smoothing
    b_yes = TeleopBridge(model, filter_window=5)         # 5-sample smoothing

    ee_no,  _ = run_trajectory(b_no,  noisy)
    ee_yes, _ = run_trajectory(b_yes, noisy)
    s_no, s_yes = smoothness(ee_no), smoothness(ee_yes)
    ratio = s_no / max(s_yes, 1e-12)
    detail = (f"window=1 jerk={s_no*1e6:.2f}µ   "
              f"window=5 jerk={s_yes*1e6:.2f}µ   "
              f"smoother×{ratio:.1f}")
    return TestResult("heavy_noise", ratio > 1.5, detail)


def t3_frame_mismatch(model):
    """Pika frame yawed +90° about z. Without remap, pika+x → robot+x (wrong:
    should be robot+y).  With R_remap = R(z, 90°), it should track correctly."""
    R_remap = R.from_euler("z", 90, degrees=True)
    b_no    = TeleopBridge(model)
    b_remap = TeleopBridge(model, R_pika_to_robot=R_remap)

    traj = make_line([0, 0, 0], [0.30, 0, 0], 200)
    _, tgt_no    = run_trajectory(b_no, traj)
    _, tgt_remap = run_trajectory(b_remap, traj)

    # No remap: pika +x delta → target +x delta (in robot world)
    dx_no = tgt_no[-1, 0] - tgt_no[0, 0]
    dy_no = tgt_no[-1, 1] - tgt_no[0, 1]
    # With remap (R_z(90°) maps +x → +y in world): target +y
    dx_re = tgt_remap[-1, 0] - tgt_remap[0, 0]
    dy_re = tgt_remap[-1, 1] - tgt_remap[0, 1]

    detail = (f"no remap   Δ=({dx_no*1000:+.0f},{dy_no*1000:+.0f})mm   "
              f"with remap Δ=({dx_re*1000:+.0f},{dy_re*1000:+.0f})mm")
    # With remap, target should move +30cm in y, not x
    passed = (abs(dy_re - 0.30) < 0.05) and (abs(dx_re) < 0.05)
    return TestResult("frame_mismatch", passed, detail)


def t4_scaling_mismatch(model):
    """Pika reports 1/10 of real motion. pos_scale=10 should compensate."""
    pika_traj = make_line([0, 0, 0], [0.030, 0, 0], 200)   # only 3 cm reported

    b_1  = TeleopBridge(model, pos_scale=1.0)
    b_10 = TeleopBridge(model, pos_scale=10.0)
    _, tgt_1  = run_trajectory(b_1,  pika_traj)
    _, tgt_10 = run_trajectory(b_10, pika_traj)
    dx_1  = tgt_1[-1, 0] - tgt_1[0, 0]
    dx_10 = tgt_10[-1, 0] - tgt_10[0, 0]
    detail = (f"scale=1 → target Δx={dx_1*1000:.0f}mm   "
              f"scale=10 → target Δx={dx_10*1000:.0f}mm")
    return TestResult("scaling_mismatch", abs(dx_10 - 0.30) < 0.05, detail)


def t5_starting_offset(model):
    """Pika baseline at (1.5, 0.3, 0.5). Engage must absorb so only delta matters."""
    bridge = TeleopBridge(model)
    pika_start = np.array([1.5, 0.3, 0.5])
    pika_traj = make_line(pika_start, pika_start + [0.30, 0, 0], 200)
    ee, tgt = run_trajectory(bridge, pika_traj)
    # Target should have moved 30cm in x from engage (robot's initial EE).
    dx = tgt[-1, 0] - bridge.robot_initial_ee[0]
    detail = (f"robot_initial_ee_x={bridge.robot_initial_ee[0]:.3f}  "
              f"target final x={tgt[-1, 0]:.3f}  Δ={dx*1000:.0f}mm")
    return TestResult("starting_offset", abs(dx - 0.30) < 0.05, detail)


def t6_workspace_edge_regrip(model):
    """Pika hits boundary at 20cm. Re-grip resets baseline → 2nd push accumulates."""
    bridge = TeleopBridge(model)

    # 1st push: 0 → 0.20m
    first  = make_line([0, 0, 0], [0.20, 0, 0], 100)
    run_trajectory(bridge, first)
    target_after_first = bridge.target_pos.copy()

    # Re-grip: operator pulls pika back; bridge clears baseline
    bridge.reset_baseline()

    # 2nd push: also 0 → 0.20m
    second = make_line([0, 0, 0], [0.20, 0, 0], 100)
    run_trajectory(bridge, second)
    target_after_second = bridge.target_pos.copy()

    total_delta = target_after_second[0] - bridge.robot_initial_ee[0]
    detail = (f"after 1st push: target x={target_after_first[0]:.3f}  "
              f"after re-grip + 2nd: target x={target_after_second[0]:.3f}  "
              f"total Δ from engage={total_delta*1000:.0f}mm")
    # Expect ~0.40m total movement (2× 20cm)
    return TestResult("workspace_edge_regrip", total_delta > 0.30, detail)


def t7_extreme_noise(model):
    """5cm σ noise (way beyond realistic).  EE jerk must still be lower with
    window=5 than window=1, otherwise the filter is failing to track signal."""
    rng = np.random.default_rng(99)
    base = make_line([0, 0, 0], [0.30, 0, 0], 200)
    noisy = base + rng.normal(0, 0.05, base.shape)
    b1 = TeleopBridge(model, filter_window=1)
    b5 = TeleopBridge(model, filter_window=5)
    ee1, _ = run_trajectory(b1, noisy)
    ee5, _ = run_trajectory(b5, noisy)
    s1, s5 = smoothness(ee1), smoothness(ee5)
    detail = f"window=1 jerk={s1*1e6:.1f}µ  window=5 jerk={s5*1e6:.1f}µ  ratio×{s1/max(s5,1e-12):.1f}"
    return TestResult("extreme_noise", s5 < s1, detail)


def t8_idle_drift(model):
    """Pika stationary but with noise.  Target should NOT drift over time."""
    rng = np.random.default_rng(0)
    n = 600   # 20 sec at 30Hz
    pika = np.tile([0.0, 0.0, 0.0], (n, 1)) + rng.normal(0, 0.005, (n, 3))
    bridge = TeleopBridge(model, filter_window=5)
    _, tgt = run_trajectory(bridge, pika)
    # Target should have moved less than 5cm despite 20s of noise
    total_drift = np.linalg.norm(tgt[-1] - tgt[0])
    detail = f"target drift over 20s of pika idle+noise = {total_drift*1000:.1f}mm"
    return TestResult("idle_drift", total_drift < 0.05, detail)


def t9_pika_upside_down(model):
    """Operator holding pika upside down (180° roll).  With auto remap
    (R_pika_to_robot = R_robot_initial * R_pika_engage.inv()), should still
    feel intuitive: pika +x in operator's hand → robot +x in its engage body."""
    # Simulate pika held upside down (roll 180°): pika +y, +z are flipped
    pika_engage_rot = R.from_euler("x", 180, degrees=True)
    # If robot engage rot is identity (it's not, but assume), R_remap = pika_engage_rot.inv()
    # In the real sim, R_remap is built at engage from actual robot rot * pika.inv().
    bridge = TeleopBridge(model)        # raw delta, no remap
    bridge_auto = None                   # we'll compute after engage

    quat_eng = pika_engage_rot.as_quat()
    # Operator pushes pika in pika's body +x (= world -x because of 180° roll about x...
    # wait, 180° about x flips y,z but leaves x. So pika body +x = world +x.
    # Use roll about z by 180° instead to make +x actually different.
    pika_engage_rot = R.from_euler("z", 180, degrees=True)
    quat_eng = pika_engage_rot.as_quat()
    # body +x of this rotated pika is world -x.  Push pika "in operator's forward"
    # means pika body +x direction → world -x.
    pika_traj = make_line([0, 0, 0], [-0.30, 0, 0], 200)   # pika in world -x
    quat_traj = [quat_eng] * 200
    _, tgt_no = run_trajectory(bridge, pika_traj, quat_traj=quat_traj)
    # Without remap, target moves -x.  With auto remap on real system, it would
    # be transformed to robot body forward.  For this test, we just check raw
    # behaviour without remap is "wrong" (-x), and apply explicit R_z(180°) remap
    # to show fix.
    R_remap = R.from_euler("z", 180, degrees=True)
    bridge_remap = TeleopBridge(model, R_pika_to_robot=R_remap)
    _, tgt_re = run_trajectory(bridge_remap, pika_traj, quat_traj=quat_traj)
    dx_no = tgt_no[-1, 0] - tgt_no[0, 0]
    dx_re = tgt_re[-1, 0] - tgt_re[0, 0]
    detail = f"no remap target Δx={dx_no*1000:+.0f}mm   with R_z(180°) Δx={dx_re*1000:+.0f}mm"
    # Without remap: -300mm.  With remap: +300mm.
    return TestResult("pika_upside_down", dx_re > 0.20, detail)


def t10_combined(model):
    """All four pathologies at once: frame yaw 90°, scale 5×, big offset, heavy noise.
    With proper remap + scale + filter, EE delta should still be in the right direction
    (robot +y) and magnitudes roughly correct."""
    rng = np.random.default_rng(7)
    R_remap = R.from_euler("z", 90, degrees=True)
    bridge = TeleopBridge(model, pos_scale=5.0, filter_window=5,
                            R_pika_to_robot=R_remap)

    # Pika reports 0.2 / 5 = 4cm of motion (operator moved 20cm physically)
    pika_start = np.array([2.5, -0.7, 1.2])               # big offset
    pika_end = pika_start + np.array([0.040, 0, 0])       # tiny reported motion
    base = make_line(pika_start, pika_end, 200)
    noisy = base + rng.normal(0, 0.005, base.shape)       # 5mm noise

    _, tgt = run_trajectory(bridge, noisy)
    dx = tgt[-1, 0] - bridge.robot_initial_ee[0]
    dy = tgt[-1, 1] - bridge.robot_initial_ee[1]
    # Expected: +20cm in robot +y (after pos_scale × 5 of reported 4cm = 20cm,
    # then yaw-90° remap maps +x → +y)
    detail = (f"target Δ from engage: Δx={dx*1000:+.0f}mm Δy={dy*1000:+.0f}mm  "
              f"(expected ≈ +0mm x, +200mm y)")
    passed = (abs(dy - 0.20) < 0.05) and (abs(dx) < 0.05)
    return TestResult("combined", passed, detail)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                    help="MJCF (defaults to mobile_manipulator.xml next to this file)")
    args = p.parse_args()

    here = Path(__file__).parent
    os.chdir(here)
    model = args.model or str(here / "mobile_manipulator.xml")
    if not Path(model).exists():
        print(f"FATAL: model {model!r} not found.")
        sys.exit(1)
    print(f"Model: {model}")
    print(f"Working dir: {os.getcwd()}\n")

    tests = [t1_baseline, t2_heavy_noise, t3_frame_mismatch,
             t4_scaling_mismatch, t5_starting_offset,
             t6_workspace_edge_regrip, t7_extreme_noise,
             t8_idle_drift, t9_pika_upside_down, t10_combined]

    results: list[TestResult] = []
    for fn in tests:
        print(f"--- {fn.__name__} ---")
        try:
            r = fn(model)
            print(f"  {r.detail}")
            print(f"  → {'PASS ✓' if r.passed else 'FAIL ✗'}\n")
            results.append(r)
        except Exception as e:
            traceback.print_exc()
            results.append(TestResult(fn.__name__, False, f"EXCEPTION: {e}"))

    print("=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for r in results if r.passed)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  {r.name:30s} {mark}   {r.detail}")
    print(f"\n  {n_pass}/{len(results)} passed")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
