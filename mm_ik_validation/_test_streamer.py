"""Local test script for two-tier streaming + gripper."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from whole_body_ik import WholeBodyIK, rpy_to_quat_wxyz
from whole_body_executor import StreamingController, SimBaseIO, SimArmIO


def run_scenario(label, target_fn, streamer, ik, base_io, arm_io,
                  n_ticks, dt, gripper_fn=None, print_every=15):
    print()
    print(f"=== {label} ===")
    print("t      EE_err   base_pos                    tier  gripper")
    print("-" * 80)
    quat = rpy_to_quat_wxyz(0, 1.5708, 0)
    for i in range(n_ticks):
        t = i * dt
        target_pos = target_fn(t)
        streamer.set_target(target_pos, quat)
        if gripper_fn:
            streamer.set_gripper(gripper_fn(t))
        streamer.tick(dt)
        if i % print_every == 0:
            cur_ee, _ = ik.get_ee_pose()
            err_mm = float(np.linalg.norm(cur_ee - target_pos) * 1000)
            cp = base_io.get_pose()
            g = arm_io.get_gripper()
            print(f"{t:5.2f}  {err_mm:6.1f}mm  ({cp[0]:+.3f},{cp[1]:+.3f},{np.degrees(cp[2]):+5.1f}°)   T{streamer.last_tier_used}    {g*1000:.1f}mm")
    s = streamer.tier_stats()
    total = s["tier1_count"] + s["tier2_count"]
    pct = s["tier1_pct"]
    print(f"-> tier1 used {s['tier1_count']}/{total} = {pct:.0f}%   "
          f"(base_moved={np.hypot(base_io.get_pose()[0], base_io.get_pose()[1])*1000:.0f}mm)")


def main():
    ik = WholeBodyIK("mobile_manipulator.xml")
    base_io = SimBaseIO(ik)
    arm_io = SimArmIO(ik)
    home_ee, _ = ik.get_ee_pose()
    print(f"Home EE: ({home_ee[0]:.3f}, {home_ee[1]:.3f}, {home_ee[2]:.3f})")

    streamer = StreamingController(ik, base_io, arm_io)
    print(f"defaults: tier1={streamer.tier1_cost} tier2={streamer.tier2_cost} "
          f"thr={streamer.fallback_pos_threshold*1000:.0f}mm")

    dt = 1/30

    # Test 1: within arm reach (should stay TIER 1 = arm only)
    run_scenario(
        "Scenario A: target stays within arm reach (slow ramp 10cm)",
        target_fn=lambda t: np.array([home_ee[0] + min(t/2.0, 1.0) * 0.10,
                                        home_ee[1] + min(t/2.0, 1.0) * 0.05,
                                        home_ee[2] - min(t/2.0, 1.0) * 0.10]),
        gripper_fn=lambda t: 0.0 if t < 1.5 else 0.035,
        streamer=streamer, ik=ik, base_io=base_io, arm_io=arm_io,
        n_ticks=90, dt=dt)

    # Reset and test 2
    ik.reset_home()
    streamer2 = StreamingController(ik, base_io, arm_io)
    run_scenario(
        "Scenario B: target far away (1m) — must trigger TIER 2",
        target_fn=lambda t: np.array([home_ee[0] + 1.0, home_ee[1], home_ee[2] - 0.3]),
        streamer=streamer2, ik=ik, base_io=base_io, arm_io=arm_io,
        n_ticks=150, dt=dt, print_every=30)

    # Reset and test 3: force arm-only mode
    ik.reset_home()
    streamer3 = StreamingController(ik, base_io, arm_io, force_arm_only=True)
    run_scenario(
        "Scenario C: same far target but force_arm_only=True (base must NOT move)",
        target_fn=lambda t: np.array([home_ee[0] + 1.0, home_ee[1], home_ee[2] - 0.3]),
        streamer=streamer3, ik=ik, base_io=base_io, arm_io=arm_io,
        n_ticks=90, dt=dt, print_every=30)


if __name__ == "__main__":
    main()
