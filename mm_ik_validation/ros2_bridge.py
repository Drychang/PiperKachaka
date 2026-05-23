"""
ros2_bridge.py

ROS2 bridge for the whole-body executor.

Architecture (same as the sim demo, just swap the I/O adapters):

       /whole_body/goal  (PoseStamped, world frame)
              │
              ▼
       WholeBodyIKNode
       ┌────────────────────────────────────┐
       │  WholeBodyExecutor (unchanged)     │
       │    DiffDriveController             │
       │    ArmInterpolator                 │
       └────┬───────────────────────────┬───┘
            │ Ros2BaseIO                │ Ros2ArmIO
            ▼                           ▼
       /odom (sub)                /joint_states (sub)
       /cmd_vel (pub)             <ARM_CMD_TOPIC> (pub)

Run (on the machine with ROS2 + the piper/kachaka topics):

    source /opt/ros/<distro>/setup.bash
    source <your_ws>/install/setup.bash
    python ros2_bridge.py

Test (publish a 6D pose target):

    ros2 topic pub --once /whole_body/goal geometry_msgs/PoseStamped \
        '{header: {frame_id: "map"}, pose: {position: {x: 0.5, y: 0.3, z: 0.8}, \
        orientation: {w: 0.7071068, x: 0.0, y: 0.7071068, z: 0.0}}}'

You MUST verify and adjust:
    ODOM_TOPIC, CMD_VEL_TOPIC, JOINT_STATES_TOPIC, ARM_CMD_TOPIC, ARM_CMD_MSG_TYPE
    ARM_JOINT_NAMES, BASE_FRAME, MAP_FRAME
to match your actual Kachaka + Piper ROS2 interfaces.

The first time you run this, set DRY_RUN = True so nothing actually moves
on the real robot — it just prints "I would publish ... ".
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path

import numpy as np

# Allow importing from this directory
sys.path.insert(0, str(Path(__file__).parent))

from whole_body_ik import WholeBodyIK, rpy_to_quat_wxyz                  # noqa: E402
from whole_body_executor import (                                          # noqa: E402
    BaseIO, ArmIO,
    DiffDriveController, ArmInterpolator,
    WholeBodyExecutor, Phase,
    StreamingController,
)

# ====================== USER CONFIG — EDIT THESE ============================
DRY_RUN              = True                       # set False only after verification
MODE                 = "stream"                   # "goto" or "stream"
MODEL_PATH           = "mobile_manipulator.xml"
ODOM_TOPIC           = "/odom"
CMD_VEL_TOPIC        = "/cmd_vel"
JOINT_STATES_TOPIC   = "/joint_states"            # Piper joints (subscribe AND publish — see notes)
ARM_CMD_TOPIC        = "/joint_states"            # Piper accepts JointState here as a command
GOAL_TOPIC           = "/whole_body/goal"         # GOTO mode: one-shot 6D pose
STREAM_TOPIC         = "/whole_body/target_pose"  # STREAM mode: continuous pose feed
GRIPPER_CMD_TOPIC    = "/whole_body/gripper_cmd"  # external gripper command (Float64, meters)
STATUS_TOPIC         = "/whole_body/status"
ARM_JOINT_NAMES      = ["joint1", "joint2", "joint3",
                        "joint4", "joint5", "joint6"]
CONTROL_RATE_HZ      = 30.0
STREAM_TIMEOUT_S     = 0.5                        # stop base if no pose received in this window
# Two-tier streaming behavior — see whole_body_executor.StreamingController docstring
TIER1_BASE_DAMPING   = 500.0                      # arm-only damping (very high = base nearly fixed)
TIER2_BASE_DAMPING   = 5.0                        # fallback damping (base allowed)
FALLBACK_POS_THRESH  = 0.05                       # m, switch to tier 2 if EE pos error > this
FALLBACK_ROT_THRESH  = 0.10                       # rad, switch to tier 2 if EE rot error > this
FORCE_ARM_ONLY       = False                      # if True, base will NEVER move (even if EE can't reach)
# ============================================================================

import rclpy                                                              # noqa: E402
from rclpy.node import Node                                               # noqa: E402
from geometry_msgs.msg import Twist, PoseStamped                          # noqa: E402
from nav_msgs.msg import Odometry                                         # noqa: E402
from sensor_msgs.msg import JointState                                    # noqa: E402
from std_msgs.msg import String, Float64                                  # noqa: E402


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class Ros2BaseIO(BaseIO):
    """Reads /odom for current 2D pose; publishes /cmd_vel for diff drive."""
    def __init__(self, node: Node, dry_run: bool):
        self.node = node
        self.dry_run = dry_run
        self._pose = np.array([0.0, 0.0, 0.0])
        self._got_odom = False
        node.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, 10)
        self.pub = node.create_publisher(Twist, CMD_VEL_TOPIC, 10)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._pose = np.array([p.x, p.y, quat_to_yaw(o.x, o.y, o.z, o.w)])
        self._got_odom = True

    def get_pose(self):
        if not self._got_odom:
            self.node.get_logger().warn(f"No odom received yet on {ODOM_TOPIC}")
        return self._pose.copy()

    def send_velocity(self, v, w, dt):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        if self.dry_run:
            if abs(v) > 1e-4 or abs(w) > 1e-4:
                self.node.get_logger().info(
                    f"[DRY_RUN] cmd_vel: v={v:+.3f} w={w:+.3f}")
        else:
            self.pub.publish(msg)


class Ros2ArmIO(ArmIO):
    """Reads /joint_states for current arm joints + gripper; publishes targets."""
    GRIPPER_JOINT_NAME = "gripper"   # the joint name your piper driver uses

    def __init__(self, node: Node, dry_run: bool):
        self.node = node
        self.dry_run = dry_run
        self._q = np.zeros(len(ARM_JOINT_NAMES))
        self._gripper = 0.0
        self._got_state = False
        self._last_cmd = None
        node.create_subscription(JointState, JOINT_STATES_TOPIC,
                                  self._on_joint_states, 10)
        self.pub = node.create_publisher(JointState, ARM_CMD_TOPIC, 10)

    def _on_joint_states(self, msg: JointState):
        # Map by name in case the published order differs.
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            self._q = np.array([name_to_pos[n] for n in ARM_JOINT_NAMES])
            self._got_state = True
        except KeyError as e:
            self.node.get_logger().warn_once(
                f"joint {e} not in /joint_states; names received: {list(msg.name)}")
        # Gripper is optional (kept at last known if absent)
        if self.GRIPPER_JOINT_NAME in name_to_pos:
            self._gripper = float(name_to_pos[self.GRIPPER_JOINT_NAME])

    def get_joints(self):
        if not self._got_state:
            self.node.get_logger().warn(f"No joint state received yet on {JOINT_STATES_TOPIC}")
        return self._q.copy()

    def get_gripper(self):
        return self._gripper

    def send_joints(self, q, gripper: float | None = None):
        # If gripper is None, keep the last commanded (or sensed) value so that
        # the published JointState always contains a valid "gripper" entry —
        # piper driver expects all 7 positions to be present.
        if gripper is None:
            gripper = self._gripper
        else:
            gripper = float(np.clip(gripper, self.GRIPPER_MIN, self.GRIPPER_MAX))

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(ARM_JOINT_NAMES) + [self.GRIPPER_JOINT_NAME]
        msg.position = [float(qi) for qi in q] + [float(gripper)]
        if self.dry_run:
            arm_changed = (self._last_cmd is None or
                            np.linalg.norm(q - self._last_cmd) > 0.05)
            grip_changed = (not hasattr(self, "_last_grip_cmd") or
                             self._last_grip_cmd is None or
                             abs(gripper - self._last_grip_cmd) > 0.005)
            if arm_changed or grip_changed:
                self.node.get_logger().info(
                    f"[DRY_RUN] arm cmd: {[round(float(qi), 3) for qi in q]}  "
                    f"gripper={gripper:.3f}")
        else:
            self.pub.publish(msg)
        self._last_cmd = q.copy()
        self._last_grip_cmd = gripper


class WholeBodyIKNode(Node):
    def __init__(self):
        super().__init__("whole_body_ik_bridge")
        self.ik = WholeBodyIK(MODEL_PATH)
        self.base_io = Ros2BaseIO(self, DRY_RUN)
        self.arm_io  = Ros2ArmIO(self, DRY_RUN)

        self.mode = MODE
        self._last_stream_msg_time = None

        if self.mode == "goto":
            self.executor = WholeBodyExecutor(
                self.ik, self.base_io, self.arm_io,
                base_ctrl=DiffDriveController(max_v=0.2, max_w=0.6,
                                                pos_tol=0.03, yaw_tol_final=0.05),
                arm_interp=ArmInterpolator(duration=3.0),
            )
            self.goal_sub = self.create_subscription(
                PoseStamped, GOAL_TOPIC, self._on_goal, 1)
            input_desc = f"PUB ONCE to {GOAL_TOPIC} (PoseStamped) to trigger"
        elif self.mode == "stream":
            self.streamer = StreamingController(
                self.ik, self.base_io, self.arm_io,
                max_v=0.2, max_w=0.6,
                yaw_assist_gain=2.0,
                inner_iters=5,
                tier1_base_damping=TIER1_BASE_DAMPING,
                tier2_base_damping=TIER2_BASE_DAMPING,
                fallback_pos_threshold=FALLBACK_POS_THRESH,
                fallback_rot_threshold=FALLBACK_ROT_THRESH,
                force_arm_only=FORCE_ARM_ONLY,
            )
            self.stream_sub = self.create_subscription(
                PoseStamped, STREAM_TOPIC, self._on_stream, 1)
            input_desc = f"PUB CONTINUOUSLY to {STREAM_TOPIC} (PoseStamped)"
        else:
            raise ValueError(f"Unknown MODE: {self.mode}")

        # Gripper subscription — works in both modes
        self.gripper_sub = self.create_subscription(
            Float64, GRIPPER_CMD_TOPIC, self._on_gripper, 1)
        self._gripper_target: float | None = None

        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
        self.dt = 1.0 / CONTROL_RATE_HZ
        self.timer = self.create_timer(self.dt, self._tick)
        self._last_phase = None
        self.get_logger().info(
            f"Bridge ready.  MODE={self.mode}  DRY_RUN={DRY_RUN}.\n  {input_desc}")

    def _on_goal(self, msg: PoseStamped):
        if self.executor.phase != Phase.IDLE:
            self.get_logger().warn("Already executing, ignoring new goal")
            return
        # We treat goal frame as the same as base odom frame (world). Caller
        # is responsible for sending poses in the correct frame.
        p = msg.pose.position
        o = msg.pose.orientation
        target_pos = np.array([p.x, p.y, p.z])
        target_quat = np.array([o.w, o.x, o.y, o.z])
        self.get_logger().info(
            f"Goal received: pos={target_pos.tolist()} quat_wxyz={target_quat.tolist()}")
        # Use current robot state as IK initial guess.
        # qpos order: [base_x, base_y, base_yaw, joint1..joint6, joint7, joint8]
        base = self.base_io.get_pose()
        arm  = self.arm_io.get_joints()
        # write current state into ik.data.qpos before calling executor.begin
        addrs = self.ik.ik_qpos_addr
        self.ik.data.qpos[addrs[0]] = base[0]
        self.ik.data.qpos[addrs[1]] = base[1]
        self.ik.data.qpos[addrs[2]] = base[2]
        for i in range(6):
            self.ik.data.qpos[addrs[3 + i]] = arm[i]
        self.ik.configuration.update(self.ik.data.qpos)
        ok = self.executor.begin(target_pos, target_quat)
        if ok:
            self.get_logger().info(
                f"IK ok. base_target=({self.executor.base_target[0]:+.3f}, "
                f"{self.executor.base_target[1]:+.3f}, "
                f"{math.degrees(self.executor.base_target[2]):+.1f}deg)")
        else:
            self.get_logger().error(f"IK failed: {self.executor.last_error}")

    def _on_stream(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        quat = np.array([o.w, o.x, o.y, o.z])
        self.streamer.set_target(pos, quat)
        self._last_stream_msg_time = time.time()

    def _on_gripper(self, msg: Float64):
        """
        Set gripper target. Value is in meters of opening:
            0.0   = closed
            0.035 = fully open
        Hardware may use different units (radians, percentage, etc.) — adjust
        the clamp here or convert in your teleop publisher.
        """
        self._gripper_target = float(msg.data)
        if self.mode == "stream":
            self.streamer.set_gripper(self._gripper_target)
        else:
            self.executor.set_gripper(self._gripper_target)
        self.get_logger().info(f"Gripper target = {self._gripper_target:.3f} m")

    def _tick(self):
        if self.mode == "goto":
            return self._tick_goto()
        elif self.mode == "stream":
            return self._tick_stream()

    def _tick_goto(self):
        s = String()
        s.data = self.executor.phase.name
        self.status_pub.publish(s)
        if self.executor.phase != self._last_phase:
            self.get_logger().info(f"Phase: {self.executor.phase.name}")
            self._last_phase = self.executor.phase

        # Always honour the latest gripper command, even when arm/base are idle.
        # This sends a JointState with current arm + new gripper, so users can
        # open/close the hand between motions.
        if (self._gripper_target is not None and
            getattr(self, "_last_pub_gripper", None) != self._gripper_target):
            cur_arm = self.arm_io.get_joints()
            self.arm_io.send_joints(cur_arm, gripper=self._gripper_target)
            self._last_pub_gripper = self._gripper_target

        if self.executor.phase == Phase.IDLE or self.executor.phase == Phase.FAILED:
            if not DRY_RUN:
                self.base_io.pub.publish(Twist())
            return
        self.executor.step(self.dt, now=time.time())

    def _tick_stream(self):
        s = String()
        s.data = "STREAMING" if self.streamer.active else "IDLE"
        self.status_pub.publish(s)
        # Safety: if no pose received recently, halt.
        if (self._last_stream_msg_time is None or
            (time.time() - self._last_stream_msg_time) > STREAM_TIMEOUT_S):
            if self.streamer.active:
                self.get_logger().warn(
                    f"No pose msg for >{STREAM_TIMEOUT_S}s on {STREAM_TOPIC} — stopping base")
                self.streamer.stop()
                if not DRY_RUN:
                    self.base_io.pub.publish(Twist())
            return
        self.streamer.tick(self.dt, now=time.time())


def main():
    rclpy.init()
    node = WholeBodyIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
