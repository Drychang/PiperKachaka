"""
whole_body_executor.py

Two-stage executor: IK -> drive base (diff drive) -> move arm.

Designed to share code between simulation and real robot:
  - Controller logic (DiffDriveController, ArmInterpolator) is pure Python.
  - Execution interface (SimBaseIO / SimArmIO) is a thin adapter that talks to
    MuJoCo data. For real robot, replace with a ROS2 adapter (cmd_vel publisher,
    joint position publisher, odom/joint_states subscribers).

Usage (sim demo):
    python whole_body_executor.py demo
        - Drag pink target to set 6D pose.
        - Press SPACE: run IK, then drive base, then move arm.
        - Press R:     reset to home.
        - Press ESC:   cancel current execution.
"""

from __future__ import annotations
import argparse
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

import mujoco
import mujoco.viewer
import numpy as np

import mink
from whole_body_ik import WholeBodyIK, rpy_to_quat_wxyz, IK_JOINT_NAMES, BASE_JOINT_NAMES


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


# --------------------------------------------------------------------------- #
# Pure-logic controllers (no sim / ROS dependencies)                          #
# --------------------------------------------------------------------------- #

@dataclass
class DiffDriveController:
    """
    Turn-drive-turn FSM for differential drive base.

    Phases:
        TURN_TO_HEADING:  rotate in place to face the target position
        DRIVE:            drive forward toward target (with small heading correction)
        TURN_TO_FINAL:    rotate in place to final yaw
        DONE

    Pure function: compute_cmd(current_xyt, target_xyt) -> (v, w, phase, done)
    Caller is responsible for stepping the base (sim integrator or ROS publisher).
    """
    max_v: float = 0.25
    max_w: float = 0.8
    v_gain: float = 1.2
    w_gain: float = 2.5
    pos_tol: float = 0.03
    yaw_tol_heading: float = 0.08
    yaw_tol_final: float = 0.04
    skip_drive_if_close: float = 0.05

    _phase: int = field(default=0, init=False)  # 0=turn1, 1=drive, 2=turn2, 3=done

    def reset(self):
        self._phase = 0

    @property
    def phase_name(self) -> str:
        return ["TURN_TO_HEADING", "DRIVE", "TURN_TO_FINAL", "DONE"][self._phase]

    def compute_cmd(self, current: np.ndarray, target: np.ndarray):
        """current/target: np.array([x, y, theta]). Returns (v, w, done)."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dist = float(np.hypot(dx, dy))

        if self._phase == 0:
            if dist < self.skip_drive_if_close:
                self._phase = 2
                return self.compute_cmd(current, target)
            heading_target = float(np.arctan2(dy, dx))
            yaw_err = wrap_to_pi(heading_target - current[2])
            if abs(yaw_err) < self.yaw_tol_heading:
                self._phase = 1
                return self.compute_cmd(current, target)
            w = float(np.clip(self.w_gain * yaw_err, -self.max_w, self.max_w))
            return 0.0, w, False

        if self._phase == 1:
            if dist < self.pos_tol:
                self._phase = 2
                return self.compute_cmd(current, target)
            heading_target = float(np.arctan2(dy, dx))
            yaw_err = wrap_to_pi(heading_target - current[2])
            v = float(np.clip(self.v_gain * dist, 0.0, self.max_v))
            w = float(np.clip(0.5 * self.w_gain * yaw_err, -self.max_w * 0.5, self.max_w * 0.5))
            return v, w, False

        if self._phase == 2:
            yaw_err = wrap_to_pi(target[2] - current[2])
            if abs(yaw_err) < self.yaw_tol_final:
                self._phase = 3
                return 0.0, 0.0, True
            w = float(np.clip(self.w_gain * yaw_err, -self.max_w, self.max_w))
            return 0.0, w, False

        return 0.0, 0.0, True


@dataclass
class ArmInterpolator:
    """Quintic ease-in-out interpolation in joint space."""
    duration: float = 2.0
    _t0: float = field(default=0.0, init=False)
    _q0: np.ndarray = field(default=None, init=False)
    _q1: np.ndarray = field(default=None, init=False)
    _active: bool = field(default=False, init=False)

    def begin(self, q_start: np.ndarray, q_target: np.ndarray, now: float):
        self._t0 = now
        self._q0 = q_start.copy()
        self._q1 = q_target.copy()
        self._active = True

    def reset(self):
        self._active = False

    def step(self, now: float):
        """Returns (q_cmd, done). q_cmd is None if not active."""
        if not self._active:
            return None, True
        alpha = (now - self._t0) / max(self.duration, 1e-6)
        if alpha >= 1.0:
            self._active = False
            return self._q1.copy(), True
        s = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
        return self._q0 + s * (self._q1 - self._q0), False


# --------------------------------------------------------------------------- #
# Execution interfaces (sim adapter — real robot will subclass these)         #
# --------------------------------------------------------------------------- #

class BaseIO(ABC):
    @abstractmethod
    def get_pose(self) -> np.ndarray: ...
    @abstractmethod
    def send_velocity(self, v: float, w: float, dt: float) -> None: ...


class ArmIO(ABC):
    @abstractmethod
    def get_joints(self) -> np.ndarray: ...
    @abstractmethod
    def send_joints(self, q: np.ndarray, gripper: float | None = None) -> None: ...
    @abstractmethod
    def get_gripper(self) -> float: ...

    # Default gripper range — override in subclass if real hardware uses different bounds.
    GRIPPER_MIN: float = 0.0     # fully closed
    GRIPPER_MAX: float = 0.035   # fully open (matches MJCF joint7 range)


class SimBaseIO(BaseIO):
    """Kinematically integrate (v, w) into MuJoCo base qpos. No physics."""
    def __init__(self, ik: WholeBodyIK):
        self.ik = ik
        self.x_addr   = int(ik.ik_qpos_addr[0])
        self.y_addr   = int(ik.ik_qpos_addr[1])
        self.yaw_addr = int(ik.ik_qpos_addr[2])

    def get_pose(self):
        return np.array([
            self.ik.data.qpos[self.x_addr],
            self.ik.data.qpos[self.y_addr],
            self.ik.data.qpos[self.yaw_addr],
        ])

    def send_velocity(self, v, w, dt):
        theta = self.ik.data.qpos[self.yaw_addr]
        self.ik.data.qpos[self.x_addr]   += v * np.cos(theta) * dt
        self.ik.data.qpos[self.y_addr]   += v * np.sin(theta) * dt
        self.ik.data.qpos[self.yaw_addr] += w * dt
        mujoco.mj_forward(self.ik.model, self.ik.data)


class SimArmIO(ArmIO):
    """Directly write 6 arm joint qpos + gripper qpos in the MuJoCo data."""
    def __init__(self, ik: WholeBodyIK):
        self.ik = ik
        self.arm_addrs = [int(ik.ik_qpos_addr[3 + i]) for i in range(6)]
        # joint7 controls the finger slide; joint8 mirrors via mimic equality.
        j7_id = mujoco.mj_name2id(ik.model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
        self.gripper_qpos_addr = int(ik.model.jnt_qposadr[j7_id])

    def get_joints(self):
        return np.array([self.ik.data.qpos[a] for a in self.arm_addrs])

    def get_gripper(self):
        return float(self.ik.data.qpos[self.gripper_qpos_addr])

    def send_joints(self, q, gripper: float | None = None):
        for a, qi in zip(self.arm_addrs, q):
            self.ik.data.qpos[a] = qi
        if gripper is not None:
            self.ik.data.qpos[self.gripper_qpos_addr] = float(
                np.clip(gripper, self.GRIPPER_MIN, self.GRIPPER_MAX))
        mujoco.mj_forward(self.ik.model, self.ik.data)


# --------------------------------------------------------------------------- #
# Top-level executor                                                          #
# --------------------------------------------------------------------------- #

class Phase(Enum):
    IDLE = auto()
    DRIVE_BASE = auto()
    MOVE_ARM = auto()
    FAILED = auto()


class WholeBodyExecutor:
    """
    Pipeline:  6D pose target
              -> IK (mink, arm-prefer)
              -> drive base via diff drive controller
              -> smoothly interpolate arm joints
    """

    def __init__(self, ik: WholeBodyIK, base_io: BaseIO, arm_io: ArmIO,
                 base_ctrl: DiffDriveController | None = None,
                 arm_interp: ArmInterpolator | None = None):
        self.ik = ik
        self.base_io = base_io
        self.arm_io = arm_io
        self.base_ctrl = base_ctrl or DiffDriveController()
        self.arm_interp = arm_interp or ArmInterpolator(duration=2.0)
        self.phase = Phase.IDLE
        self.base_target = None
        self.arm_target = None
        self.target_pos = None
        self.target_quat = None
        self.gripper_target: float | None = None
        self.last_error = ""

    def set_gripper(self, value: float | None):
        """Set gripper target. None = leave alone; otherwise clipped to range."""
        if value is None:
            self.gripper_target = None
        else:
            self.gripper_target = float(np.clip(
                value, self.arm_io.GRIPPER_MIN, self.arm_io.GRIPPER_MAX))

    def begin(self, target_pos, target_quat_wxyz):
        """Run IK from current robot state. On success, kick off the base drive."""
        saved_q = self.ik.configuration.q.copy()
        self.ik.configuration.update(saved_q)
        ok, n_iter, err = self.ik.solve_with_restart(target_pos, target_quat_wxyz,
                                                       max_iter=200, verbose=False)
        if not ok:
            self.phase = Phase.FAILED
            self.last_error = (f"IK did not converge (pos_err={np.linalg.norm(err[:3])*1000:.1f}mm, "
                               f"rot_err={np.degrees(np.linalg.norm(err[3:])):.1f}deg)")
            self.ik.configuration.update(saved_q)
            self.ik.data.qpos[:] = saved_q
            mujoco.mj_forward(self.ik.model, self.ik.data)
            return False
        sol = self.ik.get_solution()
        self.base_target = np.array([sol["base_x"], sol["base_y"], sol["base_yaw"]])
        self.arm_target = np.array([sol[f"joint{i}"] for i in range(1, 7)])
        self.target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self.target_quat = np.asarray(target_quat_wxyz, dtype=np.float64).copy()
        # Restore current state. Real execution drives toward target gradually.
        self.ik.configuration.update(saved_q)
        self.ik.data.qpos[:] = saved_q
        mujoco.mj_forward(self.ik.model, self.ik.data)
        self.base_ctrl.reset()
        self.arm_interp.reset()
        self.phase = Phase.DRIVE_BASE
        self.last_error = ""
        return True

    def _retarget_arm_after_base(self):
        """
        After base settles within tolerance, re-run IK from the actual current
        state. Base damping (100x) naturally keeps base near current pose, so
        the QP mainly refines arm joints to reach the EE target exactly.
        Returns refined arm joint values, or self.arm_target as fallback.
        """
        if self.target_pos is None:
            return self.arm_target
        # Sync configuration to data (data has true post-drive base pose)
        self.ik.configuration.update(self.ik.data.qpos)
        saved_q = self.ik.configuration.q.copy()
        ok, _, _ = self.ik.solve_with_restart(self.target_pos, self.target_quat,
                                                max_iter=200, verbose=False)
        if not ok:
            self.ik.configuration.update(saved_q)
            self.ik.data.qpos[:] = saved_q
            mujoco.mj_forward(self.ik.model, self.ik.data)
            return self.arm_target
        sol = self.ik.get_solution()
        refined = np.array([sol[f"joint{i}"] for i in range(1, 7)])
        # Restore state — caller will animate arm to the new target via interp.
        self.ik.configuration.update(saved_q)
        self.ik.data.qpos[:] = saved_q
        mujoco.mj_forward(self.ik.model, self.ik.data)
        return refined

    def cancel(self):
        self.phase = Phase.IDLE
        self.base_target = None
        self.arm_target = None
        self.base_ctrl.reset()
        self.arm_interp.reset()

    def step(self, dt: float, now: float | None = None) -> bool:
        """One control tick. Returns True if motion just completed."""
        if now is None:
            now = time.time()

        if self.phase == Phase.IDLE or self.phase == Phase.FAILED:
            return False

        if self.phase == Phase.DRIVE_BASE:
            cur = self.base_io.get_pose()
            v, w, done = self.base_ctrl.compute_cmd(cur, self.base_target)
            self.base_io.send_velocity(v, w, dt)
            if done:
                refined_arm = self._retarget_arm_after_base()
                cur_arm = self.arm_io.get_joints()
                self.arm_interp.begin(cur_arm, refined_arm, now)
                self.phase = Phase.MOVE_ARM
            return False

        if self.phase == Phase.MOVE_ARM:
            q_cmd, done = self.arm_interp.step(now)
            if q_cmd is not None:
                self.arm_io.send_joints(q_cmd, gripper=self.gripper_target)
            if done:
                self.phase = Phase.IDLE
                return True
            return False

        return False


# --------------------------------------------------------------------------- #
# Sim demo                                                                     #
# --------------------------------------------------------------------------- #

class StreamingController:
    """
    Continuous teleop tracking of a 6D pose stream.

    No phase machine, no "drive then arm". Each tick:
      1. Sync ik state from real/sim sensors (closed-loop on actual state).
      2. Set IK target = latest streamed pose.
      3. Run mink IK once (warm-started from current config) with damping_task
         active so arm is preferred.
      4. Extract holonomic base velocity (vx_w, vy_w, vw_w) from IK output.
      5. Project to diff drive: v_forward = vx*cos(theta) + vy*sin(theta);
         lateral component is converted to a yaw-assist contribution to omega.
      6. Send cmd_vel to base and new joint positions to arm CONCURRENTLY.

    Unlike WholeBodyExecutor (sequential goto), this does NOT wait for base to
    settle. Arm and base move at the same time. This matches the standard
    streaming-teleop pattern used by mink/homer mobile_tidybot.
    """

    def __init__(self, ik: WholeBodyIK, base_io: BaseIO, arm_io: ArmIO,
                 max_v: float = 0.25, max_w: float = 0.8,
                 yaw_assist_gain: float = 2.0,
                 add_velocity_limits: bool = True,
                 max_arm_vel: float = 2.0,
                 inner_iters: int = 5,
                 # Two-tier damping: arm-prefer with automatic fallback.
                 tier1_base_damping: float = 500.0,
                 tier2_base_damping: float = 5.0,
                 fallback_pos_threshold: float = 0.05,
                 fallback_rot_threshold: float = 0.10,
                 force_arm_only: bool = False,
                 # Lock specific arm joints (IK won't move them; they stay at
                 # whatever value they currently have).  e.g. ["joint5"]
                 locked_arm_joints: list[str] | None = None):
        self.ik = ik
        self.base_io = base_io
        self.arm_io = arm_io
        self.max_v = max_v
        self.max_w = max_w
        self.yaw_assist_gain = yaw_assist_gain
        self.inner_iters = max(1, int(inner_iters))
        self.target_pos = None
        self.target_quat = None
        self.active = False
        self.gripper_target: float | None = None     # None means "leave as-is"

        self.fallback_pos_threshold = fallback_pos_threshold
        self.fallback_rot_threshold = fallback_rot_threshold
        self.force_arm_only = force_arm_only

        # Build two damping tasks (own copies, not sharing ik.damping_task which
        # belongs to goto-mode). Tier 1 = nearly-fix base; tier 2 = base allowed.
        def _make_damping(cost_val: float) -> mink.DampingTask:
            cost = np.zeros((ik.model.nv,))
            for name in BASE_JOINT_NAMES:
                cost[ik.model.joint(name).dofadr[0]] = cost_val
            return mink.DampingTask(ik.model, cost)
        self.tier1_damping_task = _make_damping(tier1_base_damping)
        self.tier2_damping_task = _make_damping(tier2_base_damping)
        # ★ Extreme damping used when force_arm_only=True; truly locks base.
        self.locked_damping_task = _make_damping(1.0e6)
        self.tier1_cost = tier1_base_damping
        self.tier2_cost = tier2_base_damping

        # Stats for monitoring / logging
        self.last_tier_used: int = 0
        self._tier1_count = 0
        self._tier2_count = 0

        # Locked joints — IK can't change these (velocity ≈ 0)
        self.locked_arm_joints = set(locked_arm_joints or [])
        if self.locked_arm_joints:
            print(f"[Streamer] Locked arm joints (IK won't move): "
                  f"{sorted(self.locked_arm_joints)}")

        if add_velocity_limits:
            vel_dict = {
                "base_x":   max_v,
                "base_y":   max_v,
                "base_yaw": max_w,
            }
            for i in range(1, 7):
                jname = f"joint{i}"
                # ★ Near-zero velocity for locked joints → IK cannot move them.
                # mink VelocityLimit needs a positive value, so use 1e-6.
                vel_dict[jname] = 1e-6 if jname in self.locked_arm_joints else max_arm_vel
            self.velocity_limit = mink.VelocityLimit(self.ik.model, vel_dict)
            if self.velocity_limit not in self.ik.limits:
                self.ik.limits.append(self.velocity_limit)

    def set_target(self, target_pos, target_quat_wxyz):
        self.target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self.target_quat = np.asarray(target_quat_wxyz, dtype=np.float64).copy()
        self.active = True

    def set_gripper(self, value: float | None):
        """Set gripper target. None means 'do not touch'; otherwise clipped to range."""
        if value is None:
            self.gripper_target = None
        else:
            self.gripper_target = float(np.clip(
                value, self.arm_io.GRIPPER_MIN, self.arm_io.GRIPPER_MAX))

    def stop(self):
        self.active = False
        self.base_io.send_velocity(0.0, 0.0, 0.0)

    def tier_stats(self):
        total = max(1, self._tier1_count + self._tier2_count)
        return {
            "tier1_count": self._tier1_count,
            "tier2_count": self._tier2_count,
            "tier1_pct": 100.0 * self._tier1_count / total,
        }

    def _project_base_vel(self, vx_w, vy_w, vw, theta):
        """
        Convert holonomic base velocity (world frame) to diff drive (v, omega).
        Lateral motion that diff drive can't do directly is converted into
        yaw rotation so future ticks can use forward motion to recover.
        """
        v_forward = vx_w * np.cos(theta) + vy_w * np.sin(theta)
        v_lateral = -vx_w * np.sin(theta) + vy_w * np.cos(theta)
        omega = vw + self.yaw_assist_gain * v_lateral / max(0.15, abs(v_forward) + 0.1)
        v = float(np.clip(v_forward, -self.max_v, self.max_v))
        omega = float(np.clip(omega, -self.max_w, self.max_w))
        return v, omega

    def tick(self, dt: float, now: float | None = None):
        if not self.active or self.target_pos is None:
            return

        cur_base = self.base_io.get_pose()
        cur_arm  = self.arm_io.get_joints()
        addrs = self.ik.ik_qpos_addr
        # ★ Reset posture target to CURRENT config every tick so posture cost
        # doesn't pull arm back toward an old engage pose. With target == current,
        # posture only provides QP regularisation (tiebreaker) without bias.
        self.ik.data.qpos[addrs[0]] = cur_base[0]
        self.ik.data.qpos[addrs[1]] = cur_base[1]
        self.ik.data.qpos[addrs[2]] = cur_base[2]
        for i in range(6):
            self.ik.data.qpos[addrs[3 + i]] = cur_arm[i]
        self.ik.configuration.update(self.ik.data.qpos)
        mujoco.mj_forward(self.ik.model, self.ik.data)

        # ★ Reset posture target to current config so it doesn't pull arm
        # back toward an old engage state (which forced base to compensate).
        self.ik.posture_task.set_target_from_configuration(self.ik.configuration)

        self.ik.set_target_mocap(self.target_pos, self.target_quat)
        T_wt = mink.SE3.from_mocap_name(self.ik.model, self.ik.data, "target")
        self.ik.ee_task.set_target(T_wt)

        inner_dt = dt / self.inner_iters
        saved_q = self.ik.configuration.q.copy()

        def _solve_with(damping_task):
            """Run inner_iters QP solves; integrate each into configuration."""
            tasks = list(self.ik.tasks) + [damping_task]
            last = None
            for _ in range(self.inner_iters):
                try:
                    last = mink.solve_ik(
                        self.ik.configuration, tasks, inner_dt, "daqp",
                        damping=1e-3, limits=self.ik.limits)
                except Exception as e:
                    print(f"streaming solver error: {e}")
                    return None
                self.ik.configuration.integrate_inplace(last, inner_dt)
            return last

        # -------- TIER 1: arm-only (very high base damping, ≈ fix base) --------
        # If force_arm_only, use locked_damping_task (1e6) which truly prevents
        # base motion — not even tier 1's "expensive but allowed" can sneak through.
        tier1_task = (self.locked_damping_task
                       if self.force_arm_only else self.tier1_damping_task)
        last_vel = _solve_with(tier1_task)
        if last_vel is None:
            return

        # Check residual EE error after tier 1
        err = self.ik.ee_task.compute_error(self.ik.configuration)
        err_pos = float(np.linalg.norm(err[:3]))
        err_rot = float(np.linalg.norm(err[3:]))
        tier_used = 1

        # -------- TIER 2: fall back if arm-only insufficient --------
        if (not self.force_arm_only and
            (err_pos > self.fallback_pos_threshold or
             err_rot > self.fallback_rot_threshold)):
            # Reset and retry with low damping (base allowed)
            self.ik.configuration.update(saved_q)
            mujoco.mj_forward(self.ik.model, self.ik.data)
            last_vel = _solve_with(self.tier2_damping_task)
            if last_vel is None:
                return
            tier_used = 2
            self._tier2_count += 1
        else:
            self._tier1_count += 1
        self.last_tier_used = tier_used

        vx_w = float(last_vel[self.ik.ik_dof_addr[0]])
        vy_w = float(last_vel[self.ik.ik_dof_addr[1]])
        vw   = float(last_vel[self.ik.ik_dof_addr[2]])

        v_cmd, omega_cmd = self._project_base_vel(vx_w, vy_w, vw, cur_base[2])

        # ★ HARD OVERRIDE: when force_arm_only is set, zero out base velocity
        # at the OUTPUT layer regardless of what the IK computed. The damping
        # task alone is a soft cost — mink QP can still leak a small base
        # velocity. This guarantees base never moves under force_arm_only.
        if self.force_arm_only:
            v_cmd = 0.0
            omega_cmd = 0.0

        self.base_io.send_velocity(v_cmd, omega_cmd, dt)

        new_arm = np.array([self.ik.configuration.q[addrs[3 + i]] for i in range(6)])
        self.arm_io.send_joints(new_arm, gripper=self.gripper_target)


@dataclass
class StreamKeyCallback:
    running: bool = True
    reset: bool = False
    gripper_open: bool = False
    def __call__(self, key):
        if key == 257 or key == 335:
            self.running = not self.running
            print(f"  [tracking={'ON' if self.running else 'OFF'}]")
        elif key in (82, ord('R'), ord('r')):
            self.reset = True
        elif key in (71, ord('G'), ord('g')):
            self.gripper_open = not self.gripper_open
            print(f"  [gripper={'OPEN' if self.gripper_open else 'CLOSED'}]")


def cmd_stream(args):
    """Streaming demo: continuously track the mocap target."""
    ik = WholeBodyIK(args.model, base_damping_cost=args.base_damping)
    base_io = SimBaseIO(ik)
    arm_io = SimArmIO(ik)
    streamer = StreamingController(
        ik, base_io, arm_io,
        max_v=args.max_v, max_w=args.max_w,
        yaw_assist_gain=args.yaw_assist,
        tier1_base_damping=args.tier1_damping,
        tier2_base_damping=args.tier2_damping,
        fallback_pos_threshold=args.fallback_pos,
        fallback_rot_threshold=args.fallback_rot,
        force_arm_only=args.arm_only,
    )
    key_cb = StreamKeyCallback()

    print("=" * 60)
    print(" Streaming whole-body tracking")
    print("=" * 60)
    print("Drag the pink sphere — robot tracks continuously.")
    print()
    print(f"  max_v={args.max_v}m/s  max_w={args.max_w}rad/s")
    print(f"  TIER 1 base damping = {args.tier1_damping} (arm-only)")
    print(f"  TIER 2 base damping = {args.tier2_damping} (base allowed)")
    print(f"  Fallback to tier 2 when EE error > "
          f"{args.fallback_pos*1000:.0f}mm or {np.degrees(args.fallback_rot):.1f}°")
    if args.arm_only:
        print(f"  ⚠ FORCE_ARM_ONLY: tier 2 disabled — base will NEVER move")
    print()
    print("Keys:")
    print("  ENTER : toggle tracking on/off")
    print("  G     : toggle gripper OPEN / CLOSED")
    print("  R     : reset to home")
    print("=" * 60)

    dt = 1.0 / args.rate
    last_print = time.time()

    with mujoco.viewer.launch_passive(
        model=ik.model, data=ik.data,
        show_left_ui=True, show_right_ui=True,
        key_callback=key_cb,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(ik.model, viewer.cam)
        mink.move_mocap_to_frame(ik.model, ik.data, "target", "ee", "site")

        while viewer.is_running():
            tick_start = time.time()
            if key_cb.reset:
                key_cb.reset = False
                ik.reset_home()
                mink.move_mocap_to_frame(ik.model, ik.data, "target", "ee", "site")
                streamer.stop()

            target_pos = ik.data.mocap_pos[ik.target_mocap_id].copy()
            target_quat = ik.data.mocap_quat[ik.target_mocap_id].copy()
            streamer.set_target(target_pos, target_quat)
            streamer.set_gripper(arm_io.GRIPPER_MAX if key_cb.gripper_open
                                  else arm_io.GRIPPER_MIN)

            if key_cb.running:
                streamer.tick(dt)
            else:
                base_io.send_velocity(0.0, 0.0, dt)

            viewer.sync()

            now = time.time()
            if now - last_print > 0.5:
                cur_ee, _ = ik.get_ee_pose()
                err_mm = float(np.linalg.norm(cur_ee - target_pos) * 1000)
                cur_base = base_io.get_pose()
                cur_gripper = arm_io.get_gripper()
                tag = "ON" if key_cb.running else "OFF"
                tier_tag = f"T{streamer.last_tier_used}"
                stats = streamer.tier_stats()
                grip_tag = "OPEN" if key_cb.gripper_open else "CLOSED"
                print(f"[stream {tag}|{tier_tag}|grip={grip_tag}({cur_gripper*1000:.0f}mm)] "
                      f"EE_err={err_mm:6.1f}mm  "
                      f"base=({cur_base[0]:+.2f},{cur_base[1]:+.2f},"
                      f"{np.degrees(cur_base[2]):+5.1f}°)  "
                      f"tier1_use={stats['tier1_pct']:.0f}%")
                last_print = now

            elapsed = time.time() - tick_start
            time.sleep(max(0.0, dt - elapsed))


@dataclass
class DemoKeyCallback:
    execute: bool = False
    reset: bool = False
    cancel: bool = False
    pause: bool = False
    gripper_open: bool = False

    def __call__(self, key):
        if key == 32:                  # SPACE
            self.execute = True
        elif key in (82, ord('R'), ord('r')):
            self.reset = True
        elif key == 256:               # ESC
            self.cancel = True
        elif key in (80, ord('P'), ord('p')):
            self.pause = not self.pause
        elif key in (71, ord('G'), ord('g')):
            self.gripper_open = not self.gripper_open
            print(f"  [gripper={'OPEN' if self.gripper_open else 'CLOSED'}]")


def cmd_demo(args):
    ik = WholeBodyIK(args.model, base_damping_cost=args.base_damping)
    base_io = SimBaseIO(ik)
    arm_io = SimArmIO(ik)
    base_ctrl = DiffDriveController(
        max_v=args.max_v, max_w=args.max_w,
        pos_tol=args.pos_tol, yaw_tol_final=args.yaw_tol,
    )
    arm_interp = ArmInterpolator(duration=args.arm_duration)
    executor = WholeBodyExecutor(ik, base_io, arm_io, base_ctrl, arm_interp)
    key_cb = DemoKeyCallback()

    print("=" * 60)
    print(" Whole-body executor demo  (sim)")
    print("=" * 60)
    print("Drag the pink sphere to set 6D pose.")
    print("Keys:")
    print("  SPACE : execute  (IK -> diff drive base -> arm interp)")
    print("  G     : toggle gripper OPEN / CLOSED")
    print("  R     : reset to home")
    print("  ESC   : cancel current execution")
    print("  P     : pause physics step")
    print()
    print(f"Base: max_v={args.max_v}m/s  max_w={args.max_w}rad/s  pos_tol={args.pos_tol}m")
    print(f"Arm interp duration: {args.arm_duration}s")
    print("=" * 60)
    print()

    dt = 1.0 / 60.0
    last_phase = None
    last_report = time.time()
    last_gripper_state = None

    with mujoco.viewer.launch_passive(
        model=ik.model, data=ik.data,
        show_left_ui=True, show_right_ui=True,
        key_callback=key_cb,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(ik.model, viewer.cam)
        mink.move_mocap_to_frame(ik.model, ik.data, "target", "ee", "site")

        while viewer.is_running():
            tick_start = time.time()

            if key_cb.execute:
                key_cb.execute = False
                target_pos = ik.data.mocap_pos[ik.target_mocap_id].copy()
                target_quat = ik.data.mocap_quat[ik.target_mocap_id].copy()
                ok = executor.begin(target_pos, target_quat)
                if ok:
                    print(f">> Executing. Base target = "
                          f"({executor.base_target[0]:+.2f}, "
                          f"{executor.base_target[1]:+.2f}, "
                          f"{np.degrees(executor.base_target[2]):+.1f}deg)")
                else:
                    print(f">> IK FAILED: {executor.last_error}")

            if key_cb.reset:
                key_cb.reset = False
                executor.cancel()
                ik.reset_home()
                mink.move_mocap_to_frame(ik.model, ik.data, "target", "ee", "site")
                print(">> Reset to home")

            if key_cb.cancel:
                key_cb.cancel = False
                executor.cancel()
                print(">> Execution cancelled")

            # Apply gripper toggle directly to sim (independent of executor phase).
            # send_joints with gripper=value writes joint7; joint8 mirrors via mimic.
            if key_cb.gripper_open != last_gripper_state:
                desired = arm_io.GRIPPER_MAX if key_cb.gripper_open else arm_io.GRIPPER_MIN
                cur_arm = arm_io.get_joints()
                arm_io.send_joints(cur_arm, gripper=desired)
                last_gripper_state = key_cb.gripper_open

            if not key_cb.pause:
                just_done = executor.step(dt)
                if just_done:
                    cur_ee_pos, _ = ik.get_ee_pose()
                    target_pos = ik.data.mocap_pos[ik.target_mocap_id]
                    err = np.linalg.norm(cur_ee_pos - target_pos) * 1000
                    print(f"<< Execution finished. EE error vs target: {err:.2f} mm")

            if executor.phase != last_phase:
                print(f"   phase: {executor.phase.name}"
                      f"{' (' + executor.base_ctrl.phase_name + ')' if executor.phase == Phase.DRIVE_BASE else ''}")
                last_phase = executor.phase

            if executor.phase == Phase.DRIVE_BASE and time.time() - last_report > 0.5:
                cur = base_io.get_pose()
                d = np.linalg.norm(cur[:2] - executor.base_target[:2])
                yaw_err = abs(wrap_to_pi(cur[2] - executor.base_target[2]))
                print(f"   base: dist={d*100:5.1f}cm  yaw_err={np.degrees(yaw_err):5.1f}deg  "
                      f"sub-phase={executor.base_ctrl.phase_name}")
                last_report = time.time()

            viewer.sync()
            elapsed = time.time() - tick_start
            time.sleep(max(0.0, dt - elapsed))


def cmd_oneshot(args):
    """Execute one pose, no GUI."""
    ik = WholeBodyIK(args.model, base_damping_cost=args.base_damping)
    base_io = SimBaseIO(ik)
    arm_io = SimArmIO(ik)
    executor = WholeBodyExecutor(ik, base_io, arm_io)

    target_pos = args.pose[:3]
    target_quat = rpy_to_quat_wxyz(*args.pose[3:])
    ik.set_target_mocap(target_pos, target_quat)

    print(f"Target: pos={target_pos.tolist()}  rpy={args.pose[3:].tolist()}")
    ok = executor.begin(target_pos, target_quat)
    if not ok:
        print(f"IK failed: {executor.last_error}")
        return

    print(f"IK ok. base_target=({executor.base_target[0]:+.3f}, "
          f"{executor.base_target[1]:+.3f}, "
          f"{np.degrees(executor.base_target[2]):+.2f}deg)")
    print("Executing...")

    dt = 1.0 / 60.0
    t0 = time.time()
    last_phase = None
    while executor.phase not in (Phase.IDLE, Phase.FAILED):
        executor.step(dt, now=time.time())
        if executor.phase != last_phase:
            print(f"  phase: {executor.phase.name}")
            last_phase = executor.phase
        if time.time() - t0 > 30.0:
            print("Timeout")
            break

    cur_ee, _ = ik.get_ee_pose()
    err = np.linalg.norm(cur_ee - target_pos) * 1000
    print(f"Done in {time.time()-t0:.2f}s.  EE pos error vs target: {err:.2f} mm")


def parse_pose(s):
    parts = [float(p) for p in s.replace(",", " ").split()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("Pose must be 6 numbers")
    return np.array(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobile_manipulator.xml")
    p.add_argument("--base-damping", type=float, default=100.0)
    p.add_argument("--max-v", type=float, default=0.25)
    p.add_argument("--max-w", type=float, default=0.8)
    p.add_argument("--pos-tol", type=float, default=0.03)
    p.add_argument("--yaw-tol", type=float, default=0.04)
    p.add_argument("--arm-duration", type=float, default=2.0)

    sub = p.add_subparsers(dest="cmd", required=True)
    pd = sub.add_parser("demo", help="Interactive sim demo (goto, sequential)")
    pd.set_defaults(func=cmd_demo)
    po = sub.add_parser("oneshot", help="Execute a single pose headless (goto)")
    po.add_argument("--pose", type=parse_pose, required=True)
    po.set_defaults(func=cmd_oneshot)
    ps = sub.add_parser("stream", help="Streaming continuous tracking (teleop)")
    ps.add_argument("--rate", type=float, default=30.0,
                     help="Control loop rate in Hz (default 30)")
    ps.add_argument("--tier1-damping", type=float, default=500.0,
                     help="Tier-1 base damping cost (high = arm-only)")
    ps.add_argument("--tier2-damping", type=float, default=5.0,
                     help="Tier-2 base damping cost (low = base allowed)")
    ps.add_argument("--fallback-pos", type=float, default=0.05,
                     help="Trigger tier 2 if pos err > this (m), default 5cm")
    ps.add_argument("--fallback-rot", type=float, default=0.10,
                     help="Trigger tier 2 if rot err > this (rad), default ~5.7°")
    ps.add_argument("--arm-only", action="store_true",
                     help="Disable tier 2 entirely — base will NEVER move")
    ps.add_argument("--yaw-assist", type=float, default=2.0,
                     help="Gain converting lateral motion to yaw rotation")
    ps.set_defaults(func=cmd_stream)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
