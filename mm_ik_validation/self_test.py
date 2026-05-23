"""
self_test.py

Headless verification of the mink-based whole-body IK.

Tests:
  1. FK round-trip: sample random reachable joint configs, do FK -> get target
     pose, reset, run IK, verify EE returns to target. Tests both math correctness
     and base-prefer behavior.
  2. Random workspace pose: arbitrary 6D poses (some unreachable).
  3. Arm vs base contribution: run same poses with full damping (stage 1 only)
     vs with damping-anneal. Shows how often arm-only is sufficient.
"""

import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from whole_body_ik import (
    WholeBodyIK, mat_to_quat_wxyz, IK_JOINT_NAMES, BASE_JOINT_NAMES,
)


def sample_joint_config(ik, rng, base_xy_range=0.5):
    q_partial = []
    for name, (lo, hi) in zip(IK_JOINT_NAMES, ik.ik_qpos_range):
        if name in ("base_x", "base_y"):
            q_partial.append(rng.uniform(-base_xy_range, base_xy_range))
        elif name == "base_yaw":
            q_partial.append(rng.uniform(-np.pi, np.pi))
        else:
            lo_use = lo if np.isfinite(lo) else -np.pi
            hi_use = hi if np.isfinite(hi) else np.pi
            margin = 0.1 * (hi_use - lo_use)
            q_partial.append(rng.uniform(lo_use + margin, hi_use - margin))
    return np.array(q_partial)


def test_fk_roundtrip(ik, n=50, seed=0):
    rng = np.random.default_rng(seed)
    successes = 0
    pos_errs, rot_errs, iters = [], [], []
    base_moves = []

    print(f"\n=== Test 1: FK round-trip ({n} samples) ===")
    t0 = time.time()
    for i in range(n):
        ik.reset_home()
        sample_q = sample_joint_config(ik, rng)
        ik.data.qpos[ik.ik_qpos_addr] = sample_q
        mujoco.mj_forward(ik.model, ik.data)
        target_pos = ik.data.site_xpos[ik.ee_site_id].copy()
        target_mat = ik.data.site_xmat[ik.ee_site_id].reshape(3, 3).copy()
        target_quat = mat_to_quat_wxyz(target_mat)

        ik.reset_home()
        success, n_iter, err = ik.solve_with_restart(
            target_pos, target_quat, max_iter=150,
            eps_pos=1e-3, eps_rot=1e-2, rng=rng)

        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        sol = ik.get_solution()
        base_moves.append(np.hypot(sol["base_x"], sol["base_y"]))
        if success:
            successes += 1
        pos_errs.append(pos_err)
        rot_errs.append(rot_err)
        iters.append(n_iter)

    elapsed = time.time() - t0
    pos_errs = np.array(pos_errs)
    rot_errs = np.array(rot_errs)
    base_moves = np.array(base_moves)
    print(f"  Success rate:    {successes}/{n} ({100*successes/n:.1f}%)")
    print(f"  Position error:  median={np.median(pos_errs)*1000:6.3f}mm  "
          f"max={pos_errs.max()*1000:6.3f}mm")
    print(f"  Rotation error:  median={np.degrees(np.median(rot_errs)):6.3f}deg  "
          f"max={np.degrees(rot_errs.max()):6.3f}deg")
    print(f"  Iterations:      median={int(np.median(iters))}  mean={np.mean(iters):.0f}")
    print(f"  Base motion:     median={np.median(base_moves)*100:.1f}cm  "
          f"max={base_moves.max()*100:.1f}cm")
    print(f"  Wall time:       {elapsed:.2f}s  ({1000*elapsed/n:.0f}ms/IK)")
    return successes / n


def sample_workspace_pose(rng):
    pos = np.array([
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(0.35, 1.1),
    ])
    rotvec = rng.standard_normal(3)
    rotvec = rotvec / np.linalg.norm(rotvec) * rng.uniform(0, np.pi)
    quat_xyzw = R.from_rotvec(rotvec).as_quat()
    return pos, np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def test_random_pose(ik, n=50, seed=1):
    rng = np.random.default_rng(seed)
    successes = 0
    pos_errs, rot_errs = [], []
    base_moves = []

    print(f"\n=== Test 2: Random workspace poses ({n} samples) ===")
    t0 = time.time()
    for i in range(n):
        ik.reset_home()
        target_pos, target_quat = sample_workspace_pose(rng)
        success, _, err = ik.solve_with_restart(
            target_pos, target_quat, max_iter=150,
            eps_pos=1e-3, eps_rot=1e-2, rng=rng)
        if success:
            successes += 1
        pos_errs.append(float(np.linalg.norm(err[:3])))
        rot_errs.append(float(np.linalg.norm(err[3:])))
        sol = ik.get_solution()
        base_moves.append(np.hypot(sol["base_x"], sol["base_y"]))

    elapsed = time.time() - t0
    pos_errs = np.array(pos_errs)
    rot_errs = np.array(rot_errs)
    base_moves = np.array(base_moves)
    print(f"  Success rate:    {successes}/{n} ({100*successes/n:.1f}%)")
    print(f"  Position error:  median={np.median(pos_errs)*1000:6.3f}mm  "
          f"max={pos_errs.max()*1000:6.3f}mm")
    print(f"  Rotation error:  median={np.degrees(np.median(rot_errs)):6.3f}deg  "
          f"max={np.degrees(rot_errs.max()):6.3f}deg")
    print(f"  Base motion:     median={np.median(base_moves)*100:.1f}cm  "
          f"max={base_moves.max()*100:.1f}cm")
    print(f"  Wall time:       {elapsed:.2f}s  ({1000*elapsed/n:.0f}ms/IK)")


def test_arm_prefer_split(ik, n=30, seed=2):
    """For random poses, check how many succeed with arm-only (stage 1) vs need base."""
    print(f"\n=== Test 3: Arm-only sufficiency ({n} samples) ===")
    rng = np.random.default_rng(seed)
    arm_only_ok = 0
    needed_base = 0
    failed = 0
    for i in range(n):
        ik.reset_home()
        target_pos, target_quat = sample_workspace_pose(rng)
        # Stage 1 only (full damping)
        ok_arm, _, _ = ik.solve(target_pos, target_quat, max_iter=200,
                                  eps_pos=1e-3, eps_rot=1e-2, with_base_damping=True)
        ik.reset_home()
        # Full anneal
        ok_full, _, _ = ik.solve_with_restart(target_pos, target_quat, max_iter=150, rng=rng)

        if ok_arm:
            arm_only_ok += 1
        elif ok_full:
            needed_base += 1
        else:
            failed += 1

    print(f"  Arm-only (stage 1 converged): {arm_only_ok}/{n} ({100*arm_only_ok/n:.0f}%)")
    print(f"  Needed base motion:           {needed_base}/{n} ({100*needed_base/n:.0f}%)")
    print(f"  Unreachable even with base:   {failed}/{n} ({100*failed/n:.0f}%)")


def main():
    print("Loading model: mobile_manipulator.xml")
    ik = WholeBodyIK("mobile_manipulator.xml")
    print(f"  nq={ik.model.nq}  nv={ik.model.nv}  nu={ik.model.nu}")
    print(f"  Base damping cost: {ik.base_damping_cost} (higher = arm preferred)")
    init_ee_pos, _ = ik.get_ee_pose()
    print(f"  Home EE pos: {init_ee_pos.tolist()}")

    test_fk_roundtrip(ik, n=50, seed=0)
    test_random_pose(ik, n=50, seed=1)
    test_arm_prefer_split(ik, n=30, seed=2)


if __name__ == "__main__":
    main()
