#!/usr/bin/env python3
"""
Run gazebo/launch/robotx_task1_uav.launch.py for a wall time, saving frames under out_clip/.

Copies the world SDF into out_clip/world.sdf for later projection labeling.

Requires ROS 2 + ros_gz_bridge on PATH (ros2 launch). Sources setup_robotx_sim_env.sh
(Gazebo paths + /opt/ros/*/setup.bash) when present, else gz_env.sh only.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    setup_env = script_dir / "setup_robotx_sim_env.sh"
    gz_env = script_dir / "gz_env.sh"
    launch = script_dir.parent / "launch" / "robotx_task1_uav.launch.py"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("world_sdf", type=Path, help="Generated RobotX world .sdf")
    p.add_argument("out_clip", type=Path, help="Directory for this clip (created)")
    p.add_argument("--seconds", type=float, default=10.0, help="Approximate recording duration once stack is up.")
    p.add_argument(
        "--startup-grace-s",
        type=float,
        default=14.0,
        help="Extra time before SIGTERM for gz + bridge + recorder spawn (matches launch TimerActions).",
    )
    p.add_argument(
        "--env-script",
        type=Path,
        default=None,
        help="Override shell script to source (default: setup_robotx_sim_env.sh if present, else gz_env.sh).",
    )
    args = p.parse_args()

    env_script = args.env_script
    if env_script is None:
        env_script = setup_env if setup_env.is_file() else gz_env
    if not env_script.is_file():
        print(f"Missing env script: {env_script}", file=sys.stderr)
        return 1
    if not launch.is_file():
        print(f"Missing {launch}", file=sys.stderr)
        return 1
    if not args.world_sdf.is_file():
        print(f"World not found: {args.world_sdf}", file=sys.stderr)
        return 1

    args.out_clip.mkdir(parents=True, exist_ok=True)
    dest_world = args.out_clip / "world.sdf"
    dest_world.write_bytes(args.world_sdf.read_bytes())

    ros2_check = (
        f'export REPO_ROOT="{repo_root}" && source "{env_script}" && command -v ros2'
    )
    ros2_ex = subprocess.run(["bash", "-lc", ros2_check], capture_output=True).returncode
    if ros2_ex != 0:
        print(
            "ros2 not on PATH after sourcing "
            f"{env_script}.\n"
            "  On Ubuntu 24.04: sudo apt install ros-jazzy-ros-gz-bridge ros-jazzy-gz-sim-vendor\n"
            f"  Then: bash -lc 'export REPO_ROOT=\"{repo_root}\" && source \"{env_script}\"'\n"
            "  See gazebo/README.md §6–8. macOS has no official ROS deb; use Linux for recording.",
            file=sys.stderr,
        )
        return 1

    total_s = max(5.0, float(args.seconds) + float(args.startup_grace_s))
    world_arg = str(args.world_sdf.resolve())
    out_arg = str(args.out_clip.resolve())

    bash_cmd = (
        f'export REPO_ROOT="{repo_root}" && source "{env_script}" && exec ros2 launch "{launch}" '
        f'world:="{world_arg}" output_dir:="{out_arg}"'
    )

    print(f"Recording ~{args.seconds}s wall time (hard cap {total_s:.1f}s) → {args.out_clip}")
    proc = subprocess.Popen(
        ["bash", "-lc", bash_cmd],
        preexec_fn=os.setsid,
        stdout=None,
        stderr=None,
    )
    try:
        proc.wait(timeout=total_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        print("Stopped launch after timeout (expected).")
    else:
        print(f"Launch exited early with code {proc.returncode}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
