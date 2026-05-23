"""
whole_body_ik.py

Whole-body IK for the Kachaka + Piper mobile manipulator, using mink
(QP-based differential IK).

Design pattern (from kevinzakka/mink `mobile_tidybot` example and
priyasundaresan/homer):

    FrameTask("ee")           # primary objective: track target pose
    PostureTask(arm only)     # regularize arm joints (tiny weight)
    DampingTask(base only)    # heavy damping on base x/y/yaw  (cost=100)

The DampingTask makes base motion "expensive" relative to arm motion, so the
QP solver prefers turning arm joints. The base only contributes when the arm
genuinely cannot reach. This solves the "base scoots around, arm never moves"
behavior of unweighted DLS IK.

Press ENTER in the viewer to toggle the damping on/off so you can compare
"base immobile" vs "base free".
"""

import argparse
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

import mink


EE_FRAME = "ee"
TARGET_BODY = "target"
IK_JOINT_NAMES = [
    "base_x", "base_y", "base_yaw",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]
BASE_JOINT_NAMES = {"base_x", "base_y", "base_yaw"}


def rpy_to_quat_wxyz(roll, pitch, yaw):
    q_xyzw = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_wxyz_to_mat(q_wxyz):
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R.from_quat(q_xyzw).as_matrix()


def mat_to_quat_wxyz(mat):
    q_xyzw = R.from_matrix(mat).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


class WholeBodyIK:
    """Mink-based whole-body IK with base-as-last-resort weighting."""

    def __init__(self, model_path,
                 base_damping_cost=100.0,
                 posture_cost_arm=1e-3,
                 pos_cost=1.0, ori_cost=1.0,
                 lm_damping=1.0):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.configuration = mink.Configuration(self.model)

        self.ee_task = mink.FrameTask(
            frame_name=EE_FRAME,
            frame_type="site",
            position_cost=pos_cost,
            orientation_cost=ori_cost,
            lm_damping=lm_damping,
        )

        posture_cost = np.zeros((self.model.nv,))
        for name in IK_JOINT_NAMES:
            if name in BASE_JOINT_NAMES:
                continue
            j_dof = self.model.joint(name).dofadr[0]
            posture_cost[j_dof] = posture_cost_arm
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)

        immobile_base_cost = np.zeros((self.model.nv,))
        for name in BASE_JOINT_NAMES:
            j_dof = self.model.joint(name).dofadr[0]
            immobile_base_cost[j_dof] = base_damping_cost
        self.damping_task = mink.DampingTask(self.model, immobile_base_cost)

        self.base_damping_cost = base_damping_cost

        self.tasks = [self.ee_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(self.model)]

        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_FRAME)

        self.ik_qpos_addr = np.array(
            [self.model.jnt_qposadr[self.model.joint(n).id] for n in IK_JOINT_NAMES], dtype=int)
        self.ik_dof_addr = np.array(
            [self.model.joint(n).dofadr[0] for n in IK_JOINT_NAMES], dtype=int)
        self.ik_qpos_range = []
        for name in IK_JOINT_NAMES:
            jid = self.model.joint(name).id
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                self.ik_qpos_range.append((float(lo), float(hi)))
            else:
                self.ik_qpos_range.append((-np.inf, np.inf))

        self.reset_home()

    def reset_home(self):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        mujoco.mj_forward(self.model, self.data)

    def set_target_mocap(self, pos, quat_wxyz):
        self.data.mocap_pos[self.target_mocap_id] = np.asarray(pos, dtype=np.float64)
        self.data.mocap_quat[self.target_mocap_id] = np.asarray(quat_wxyz, dtype=np.float64)

    def get_ee_pose(self):
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.ee_site_id].copy()
        mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return pos, mat

    def compute_error_6d(self, target_pos, target_quat_wxyz):
        cur_pos, cur_mat = self.get_ee_pose()
        pos_err = np.asarray(target_pos) - cur_pos
        target_mat = quat_wxyz_to_mat(target_quat_wxyz)
        R_err = target_mat @ cur_mat.T
        rot_err = R.from_matrix(R_err).as_rotvec()
        return np.concatenate([pos_err, rot_err])

    def solve(self, target_pos, target_quat_wxyz,
              max_iter=80, eps_pos=1e-3, eps_rot=1e-2,
              dt=0.02, with_base_damping=True, verbose=False):
        """Run mink QP IK until convergence or max_iter reached."""
        self.set_target_mocap(target_pos, target_quat_wxyz)
        mujoco.mj_forward(self.model, self.data)
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, TARGET_BODY)
        self.ee_task.set_target(T_wt)

        tasks = list(self.tasks)
        if with_base_damping:
            tasks.append(self.damping_task)

        err = np.zeros(6)
        for it in range(max_iter):
            try:
                vel = mink.solve_ik(self.configuration, tasks, dt, "daqp",
                                     damping=1e-3, limits=self.limits)
            except Exception as e:
                if verbose:
                    print(f"  iter {it}: solver error {e}")
                break
            self.configuration.integrate_inplace(vel, dt)
            self.data.qpos[:] = self.configuration.q
            mujoco.mj_forward(self.model, self.data)

            err = self.compute_error_6d(target_pos, target_quat_wxyz)
            err_pos = float(np.linalg.norm(err[:3]))
            err_rot = float(np.linalg.norm(err[3:]))
            if verbose and (it % 5 == 0 or it == max_iter - 1):
                print(f"  iter {it:3d}: pos={err_pos*1000:7.3f}mm  "
                      f"rot={np.degrees(err_rot):7.3f}deg")
            if err_pos < eps_pos and err_rot < eps_rot:
                return True, it, err

        return False, max_iter, err

    def _set_base_damping(self, cost):
        immobile_base_cost = np.zeros((self.model.nv,))
        for name in BASE_JOINT_NAMES:
            j_dof = self.model.joint(name).dofadr[0]
            immobile_base_cost[j_dof] = cost
        self.damping_task = mink.DampingTask(self.model, immobile_base_cost)

    def solve_with_restart(self, target_pos, target_quat_wxyz,
                            max_iter=80,
                            eps_pos=1e-3, eps_rot=1e-2,
                            dt=0.02, rng=None, verbose=False):
        """
        QP IK with damping-anneal restarts.

        Stage 1: full base damping  -> arm-only solution if reachable
        Stage 2: 10x lower damping  -> small base motion allowed
        Stage 3: 100x lower damping -> base moves freely
        Stage 4: no damping + random arm perturbation -> last resort

        Returns the FIRST stage that converges (which is also the one with
        the least base motion). This gives the user "arm preferred, base only
        when necessary" behavior automatically.
        """
        if rng is None:
            rng = np.random.default_rng()
        saved_q = self.configuration.q.copy()
        original_cost = self.base_damping_cost

        damping_schedule = [
            ("full damping (arm-only)",   original_cost,        False),
            ("damping/10 (mostly arm)",   original_cost * 0.1,  False),
            ("damping/100 (base allowed)", original_cost * 0.01, False),
            ("no damping + perturb",       0.0,                  True),
        ]

        best_err = np.inf
        best_q = saved_q.copy()
        best_success = False
        total_iter = 0
        stage_succeeded = None

        for stage_idx, (label, damp_cost, perturb) in enumerate(damping_schedule):
            self._set_base_damping(damp_cost)
            if perturb:
                perturbed = saved_q.copy()
                for i, name in enumerate(IK_JOINT_NAMES):
                    if name in BASE_JOINT_NAMES:
                        continue
                    qa = self.ik_qpos_addr[i]
                    lo, hi = self.ik_qpos_range[i]
                    lo_use = lo if np.isfinite(lo) else -np.pi
                    hi_use = hi if np.isfinite(hi) else np.pi
                    margin = 0.15 * (hi_use - lo_use)
                    perturbed[qa] = rng.uniform(lo_use + margin, hi_use - margin)
                    perturbed[qa] = np.clip(perturbed[qa], lo, hi)
                self.configuration.update(perturbed)
            else:
                self.configuration.update(saved_q)
            self.data.qpos[:] = self.configuration.q
            mujoco.mj_forward(self.model, self.data)

            success, n_iter, err = self.solve(
                target_pos, target_quat_wxyz,
                max_iter=max_iter, eps_pos=eps_pos, eps_rot=eps_rot,
                dt=dt, with_base_damping=True, verbose=False)
            total_iter += n_iter
            err_norm = float(np.linalg.norm(err))

            if verbose:
                print(f"  stage {stage_idx+1} [{label}]: iters={n_iter} "
                      f"pos_err={float(np.linalg.norm(err[:3]))*1000:.2f}mm "
                      f"{'CONVERGED' if success else 'failed'}")

            if err_norm < best_err:
                best_err = err_norm
                best_q = self.configuration.q.copy()
                best_success = success
            if success:
                stage_succeeded = label
                break

        self._set_base_damping(original_cost)
        self.configuration.update(best_q)
        self.data.qpos[:] = self.configuration.q
        mujoco.mj_forward(self.model, self.data)
        final_err = self.compute_error_6d(target_pos, target_quat_wxyz)
        if verbose and stage_succeeded:
            print(f"  -> converged via: {stage_succeeded}")
        return best_success, total_iter, final_err

    def get_solution(self):
        return {name: float(self.configuration.q[self.ik_qpos_addr[i]])
                for i, name in enumerate(IK_JOINT_NAMES)}


@dataclass
class KeyCallback:
    fix_base: bool = True
    pause: bool = False
    reset: bool = False
    def __call__(self, key):
        if key == 257 or key == 335:
            self.fix_base = not self.fix_base
            print(f"  [fix_base={self.fix_base}]")
        elif key == 32:
            self.pause = not self.pause
            print(f"  [pause={self.pause}]")
        elif key == 82 or key == ord('R') or key == ord('r'):
            self.reset = True


def parse_pose(s):
    parts = [float(p) for p in s.replace(",", " ").split()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("Pose must be 6 numbers: x y z roll pitch yaw")
    return np.array(parts)


def cmd_solve(args):
    ik = WholeBodyIK(args.model,
                      base_damping_cost=args.base_damping)
    target_pos = args.pose[:3]
    target_quat = rpy_to_quat_wxyz(*args.pose[3:])

    init_ee, _ = ik.get_ee_pose()
    print(f"Target pose: pos={target_pos.tolist()}, rpy={args.pose[3:].tolist()} (rad)")
    print(f"Initial EE pos: {init_ee.tolist()}")
    print(f"Base damping cost: {args.base_damping}  (higher = base more stationary)")
    print()

    if args.no_base_damping:
        print("(--no-base-damping)  Running with base free.")
        success, n_iter, err = ik.solve(target_pos, target_quat,
                                          with_base_damping=False, verbose=True,
                                          max_iter=args.max_iter)
    else:
        success, n_iter, err = ik.solve_with_restart(
            target_pos, target_quat,
            verbose=True, max_iter=args.max_iter)

    err_pos = float(np.linalg.norm(err[:3]))
    err_rot = float(np.linalg.norm(err[3:]))
    sol = ik.get_solution()
    final_ee, _ = ik.get_ee_pose()

    print()
    print(f"=== Result ===")
    print(f"Converged: {success}  (iters: {n_iter})")
    print(f"Pos err: {err_pos*1000:.3f} mm   Rot err: {np.degrees(err_rot):.3f} deg")
    print(f"Final EE pos: {final_ee.tolist()}")
    print()
    print("Solution (vs home zero for base):")
    for k, v in sol.items():
        unit = "m" if k in {"base_x", "base_y"} else "rad"
        marker = "  <- base" if k in BASE_JOINT_NAMES else ""
        print(f"  {k:10s} = {v:+8.4f} {unit}{marker}")
    print()
    print(f"Base traveled: dx={sol['base_x']:+.3f}m, dy={sol['base_y']:+.3f}m, "
          f"dyaw={np.degrees(sol['base_yaw']):+.2f}deg")


def cmd_interactive(args):
    ik = WholeBodyIK(args.model,
                      base_damping_cost=args.base_damping)
    key_cb = KeyCallback(fix_base=not args.free_base)

    print("=" * 60)
    print(" Interactive whole-body IK  (mink / QP-based)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Base damping cost: {args.base_damping}  (higher = arm preferred)")
    print(f"Posture cost on arm: {1e-3}")
    print(f"Initial mode: fix_base = {key_cb.fix_base}")
    print()
    print("In the viewer:")
    print("  - Pink sphere = mocap target (drag to set 6D pose)")
    print("  - Yellow dot at gripper = end-effector site")
    print("  - Double-click the target sphere, then Ctrl + right-drag to move.")
    print("    Ctrl + middle-drag rotates it.")
    print()
    print("Keyboard:")
    print("  ENTER : toggle fix_base (heavy base damping <-> free base)")
    print("  SPACE : pause / resume")
    print("  R     : reset to home keyframe")
    print("=" * 60)

    rate_dt = 1.0 / 60.0
    last_print = time.time()

    with mujoco.viewer.launch_passive(
        model=ik.model, data=ik.data,
        show_left_ui=True, show_right_ui=True,
        key_callback=key_cb,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(ik.model, viewer.cam)
        mink.move_mocap_to_frame(ik.model, ik.data, TARGET_BODY, EE_FRAME, "site")

        while viewer.is_running():
            if key_cb.reset:
                ik.reset_home()
                mink.move_mocap_to_frame(ik.model, ik.data, TARGET_BODY, EE_FRAME, "site")
                key_cb.reset = False

            T_wt = mink.SE3.from_mocap_name(ik.model, ik.data, TARGET_BODY)
            ik.ee_task.set_target(T_wt)

            if not key_cb.pause:
                tasks = list(ik.tasks)
                if key_cb.fix_base:
                    tasks.append(ik.damping_task)
                for _ in range(args.iter_per_frame):
                    try:
                        vel = mink.solve_ik(ik.configuration, tasks, rate_dt, "daqp",
                                             damping=1e-3, limits=ik.limits)
                    except Exception as e:
                        print(f"solver error: {e}")
                        break
                    ik.configuration.integrate_inplace(vel, rate_dt)
                ik.data.qpos[:] = ik.configuration.q
                mujoco.mj_forward(ik.model, ik.data)

            viewer.sync()

            now = time.time()
            if now - last_print > 0.5:
                target_pos = ik.data.mocap_pos[ik.target_mocap_id].copy()
                target_quat = ik.data.mocap_quat[ik.target_mocap_id].copy()
                err = ik.compute_error_6d(target_pos, target_quat)
                pos_mm = float(np.linalg.norm(err[:3]) * 1000)
                rot_deg = float(np.degrees(np.linalg.norm(err[3:])))
                sol = ik.get_solution()
                mode = "FIX_BASE" if key_cb.fix_base else "FREE_BASE"
                print(f"[{mode}] pos_err={pos_mm:6.2f}mm rot_err={rot_deg:6.2f}deg  "
                      f"base=({sol['base_x']:+.2f},{sol['base_y']:+.2f},"
                      f"{np.degrees(sol['base_yaw']):+6.1f}deg)")
                last_print = now

            time.sleep(max(0.0, rate_dt))


def main():
    parser = argparse.ArgumentParser(
        description="Whole-body IK (mink/QP) for Kachaka + Piper")
    parser.add_argument("--model", default="mobile_manipulator.xml")
    parser.add_argument("--base-damping", type=float, default=100.0,
                        help="Cost of base motion (higher = base more stationary). Default 100.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_solve = sub.add_parser("solve", help="Solve IK for a fixed 6D pose")
    p_solve.add_argument("--pose", type=parse_pose, required=True,
                         help='Target "x y z roll pitch yaw" in m/rad')
    p_solve.add_argument("--max-iter", type=int, default=200)
    p_solve.add_argument("--no-base-damping", action="store_true",
                         help="Disable base damping (base will move freely)")
    p_solve.set_defaults(func=cmd_solve)

    p_int = sub.add_parser("interactive", help="Launch viewer, drag target")
    p_int.add_argument("--iter-per-frame", type=int, default=8)
    p_int.add_argument("--free-base", action="store_true",
                       help="Start with base free instead of damped")
    p_int.set_defaults(func=cmd_interactive)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
