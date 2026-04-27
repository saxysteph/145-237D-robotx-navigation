# Gazebo Harmonic RobotX Demo (UAV Camera + Buoys)

This folder contains a Gazebo Harmonic scene for UAV-overhead buoy detection inspired by VRX finals-style layouts.

## What is included

- `gazebo/worlds/ucsd_robotx_demo.world.sdf`
  - 500m x 500m ocean plane
  - Flat ocean visual (no `gz-sim-waves-system` plugin — avoids missing-library errors on minimal installs)
  - Entrance and 3 navigation gates with emissive red/green buoys
  - Scan-the-code style light buoy target
  - 8 dark marker buoys
  - No default WAM-V include (avoids missing `model://wam-v`; add a Fuel/model URI if you need a USV)
- `gazebo/models/ucsd_drone/model.sdf`
  - Simple quadrotor geometry
  - Downward-facing camera sensor on `/drone/camera`
- `ros2_bridge/bridge.yaml`
  - Bridges `/drone/camera` and `/clock` to ROS 2
- `gazebo/launch/ucsd_demo_launch.py`
  - Starts Gazebo, spawns UAV at `60 2 25`, starts bridge, prints topic list

## Prerequisites

- Ubuntu 22.04
- ROS 2 Humble sourced
- Gazebo Harmonic tools on path (`gz`)
- `ros_gz_sim` and `ros_gz_bridge` installed
- VRX (**`jazzy` branch**) built in `~/vrx_ws` — matches **Gazebo Harmonic** (`gz-sim8`). The `humble` branch targets Garden (`gz-sim7`) and will not build with Harmonic.

## Run

From Ubuntu (or WSL Ubuntu), from the repository root:

```bash
source /opt/ros/humble/setup.bash
bash simulation/run_end_to_end.sh
```

From PowerShell on Windows:

```powershell
.\simulation\run_end_to_end.ps1
```

If you already copied files into `~/vrx_ws/src/vrx/vrx_gz` and rebuilt:

```bash
source /opt/ros/humble/setup.bash
source ~/vrx_ws/install/setup.bash
ros2 launch vrx_gz ucsd_demo_launch.py
```

## Verify camera stream

```bash
bash simulation/verify_sim_topics.sh
```

Then visualize with your preferred tool (for example `rqt_image_view` on `/drone/camera`).

**If the image is blank:** the world must load **Physics**, **SceneBroadcaster**, **UserCommands**, and **Sensors** (`gz-sim-sensors-system`) so time advances and the camera publishes. In Gazebo, press **Play** (space) so RTF is not stuck at 0%; then `ros2 topic hz /drone/camera` should show ~30 Hz.

## Wave plugin note

Animated waves were removed from the default world because `gz-sim-waves-system` is not available on all Harmonic setups (you would see `Could not find shared library`). The blue `ocean_surface` plane remains. To experiment with waves, install the full Gazebo Harmonic plugin set your distro provides, or add a third-party wave stack and update the world SDF accordingly.
