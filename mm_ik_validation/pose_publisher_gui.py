"""
pose_publisher_gui.py — simulates the pika leader controller for MuJoCo
teleop testing.

A small MuJoCo viewer with a single draggable mocap body. The mocap's 6D
pose is written to a shared-memory ring at ~30 Hz, mimicking the data
stream a real pika hand controller publishes.

Companion: mujoco_teleop_sim.py (subscribes to the same shm and drives the
kachaka + piper robot via mink IK).

Run:
    python3 pose_publisher_gui.py
    # optionally:
    #   --noise         add Gaussian jitter so subscriber filter is exercised
    #   --hz 60         publish rate
    #   --offset z=1.5  start mocap at (1.0, 0.0, 1.5) — mimics pika's "tall" frame

In the viewer:
    Double-click the pink ball, then
      Ctrl + right-drag   = translate
      Ctrl + middle-drag  = rotate

Console keys:
    g — toggle gripper open / closed
    b — send baseline reset (operator "re-grip" signal)
    r — recenter mocap to its home position
    Ctrl-C — quit

Shared memory layout (float32 × 9, name "teleop_shm"):
    [x, y, z, qx, qy, qz, qw, gripper, reset_counter]
"""
import argparse
import select
import sys
import termios
import threading
import time
import tty
from multiprocessing import shared_memory

import mujoco
import mujoco.viewer
import numpy as np


SHM_NAME       = "teleop_shm"
N_FLOATS       = 9
GRIPPER_OPEN   = 0.986
GRIPPER_CLOSED = 0.0
HOME_POS       = np.array([1.0, 0.0, 0.5], dtype=np.float64)
HOME_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

# Noise mimics real-pika tracking jitter so the subscriber's smoothing
# filter is exercised when --noise is passed.
NOISE_POS_SIGMA = 0.002        # 2 mm
NOISE_ROT_SIGMA = np.radians(0.5)


MOCAP_MJCF = """
<mujoco model="pose_publisher">
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3"/>
    <global offwidth="800" offheight="600"/>
    <scale forcewidth="0.05"/>
  </visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.4 0.4 0.4"
             rgb2="0.55 0.55 0.55" width="200" height="200"/>
    <material name="grid_mat" texture="grid" texuniform="true" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid_mat"/>

    <!-- World origin axes -->
    <site name="ox" pos="0.1 0 0.001" size="0.005 0.1" type="cylinder"
          rgba="1 0 0 0.9" euler="0 1.5708 0"/>
    <site name="oy" pos="0 0.1 0.001" size="0.005 0.1" type="cylinder"
          rgba="0 1 0 0.9" euler="-1.5708 0 0"/>
    <site name="oz" pos="0 0 0.1"     size="0.005 0.1" type="cylinder"
          rgba="0 0 1 0.9"/>

    <!-- Draggable 6D mocap target -->
    <body name="target" mocap="true" pos="1.0 0 0.5">
      <geom type="sphere" size="0.04" rgba="1 0.35 0.8 1"/>
      <site name="tx" pos="0.08 0 0" size="0.01 0.06" type="cylinder"
            rgba="1 0 0 1" euler="0 1.5708 0"/>
      <site name="ty" pos="0 0.08 0" size="0.01 0.06" type="cylinder"
            rgba="0 1 0 1" euler="-1.5708 0 0"/>
      <site name="tz" pos="0 0 0.08" size="0.01 0.06" type="cylinder"
            rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def get_key(timeout: float = 0.05) -> str:
    saved = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
    return ""


INTRO = """
============================================================
 pose_publisher_gui — virtual pika
============================================================
 In viewer:
   double-click pink ball → Ctrl + right-drag (translate)
                            Ctrl + middle-drag (rotate)

 Console keys:
   g  toggle gripper OPEN / CLOSED
   b  baseline reset  (re-grip signal to subscriber)
   r  recenter mocap to home
   Ctrl-C  quit
============================================================
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--noise", action="store_true",
                        help="add Gaussian noise to published pose")
    args = parser.parse_args()

    # Create or attach shm
    try:
        shm = shared_memory.SharedMemory(
            name=SHM_NAME, create=True, size=N_FLOATS * 4)
        owned = True
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        owned = False
    buf = np.ndarray((N_FLOATS,), dtype=np.float32, buffer=shm.buf)
    print(f"[publisher] shm '{SHM_NAME}' "
          f"{'created' if owned else 'attached (already existed)'}, "
          f"size={N_FLOATS * 4} bytes")

    # Build viewer model
    model = mujoco.MjModel.from_xml_string(MOCAP_MJCF)
    data = mujoco.MjData(model)
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    mocap_id = model.body_mocapid[target_body]
    data.mocap_pos[mocap_id] = HOME_POS
    data.mocap_quat[mocap_id] = HOME_QUAT_WXYZ
    mujoco.mj_forward(model, data)

    state = {
        "gripper":       GRIPPER_OPEN,
        "reset_counter": 0.0,
        "quit":          False,
    }

    def keyboard_loop():
        while not state["quit"]:
            k = get_key(timeout=0.05)
            if not k:
                continue
            if k == "\x03":
                state["quit"] = True
                break
            if k in ("g", "G"):
                state["gripper"] = (GRIPPER_CLOSED if state["gripper"] > 0.5
                                     else GRIPPER_OPEN)
                tag = "OPEN" if state["gripper"] > 0.5 else "CLOSED"
                print(f"\r[gripper {tag}]                              ")
            elif k in ("b", "B"):
                state["reset_counter"] += 1.0
                print(f"\r[baseline reset #{int(state['reset_counter'])} sent]")
            elif k in ("r", "R"):
                data.mocap_pos[mocap_id] = HOME_POS
                data.mocap_quat[mocap_id] = HOME_QUAT_WXYZ
                print(f"\r[mocap recentered to home]                     ")

    threading.Thread(target=keyboard_loop, daemon=True).start()

    print(INTRO)
    print(f"[publisher] publishing at {args.hz:.0f} Hz "
          f"{'(with ' + str(int(NOISE_POS_SIGMA*1000)) + 'mm pos / ' + f'{np.degrees(NOISE_ROT_SIGMA):.2f}' + '° rot noise)' if args.noise else '(no noise)'}")

    dt = 1.0 / args.hz
    rng = np.random.default_rng()
    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            while viewer.is_running() and not state["quit"]:
                t0 = time.time()
                pos = np.array(data.mocap_pos[mocap_id], copy=True)
                quat_wxyz = np.array(data.mocap_quat[mocap_id], copy=True)

                if args.noise:
                    pos = pos + rng.normal(0.0, NOISE_POS_SIGMA, size=3)
                    # small-angle noise: random axis, σ angle, compose with quat
                    axis = rng.normal(0.0, 1.0, size=3)
                    axis /= np.linalg.norm(axis) + 1e-12
                    ang = rng.normal(0.0, NOISE_ROT_SIGMA)
                    s = np.sin(ang / 2.0)
                    dq_wxyz = np.array([np.cos(ang / 2.0),
                                          s * axis[0], s * axis[1], s * axis[2]])
                    # quaternion multiply (wxyz)
                    w0, x0, y0, z0 = dq_wxyz
                    w1, x1, y1, z1 = quat_wxyz
                    quat_wxyz = np.array([
                        w0*w1 - x0*x1 - y0*y1 - z0*z1,
                        w0*x1 + x0*w1 + y0*z1 - z0*y1,
                        w0*y1 - x0*z1 + y0*w1 + z0*x1,
                        w0*z1 + x0*y1 - y0*x1 + z0*w1,
                    ])
                    quat_wxyz /= np.linalg.norm(quat_wxyz) + 1e-12

                # Write to shm:  xyz  qxyzw  gripper  reset_counter
                buf[0] = pos[0]; buf[1] = pos[1]; buf[2] = pos[2]
                buf[3] = quat_wxyz[1]      # qx
                buf[4] = quat_wxyz[2]      # qy
                buf[5] = quat_wxyz[3]      # qz
                buf[6] = quat_wxyz[0]      # qw
                buf[7] = state["gripper"]
                buf[8] = state["reset_counter"]

                viewer.sync()
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        state["quit"] = True
        print("\n[publisher] exiting.")
        try:
            shm.close()
        except Exception:
            pass
        if owned:
            try:
                shm.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
