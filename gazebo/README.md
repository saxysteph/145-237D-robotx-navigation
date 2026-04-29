# RobotX 2026 CV Dataset Simulation (ROS 2 Jazzy + Gazebo Harmonic)

This folder contains a custom Gazebo simulation setup for generating top-down UAV vision data for the RobotX "Safe Passage" style scene with:

- wave / glare-heavy ocean rendering,
- semantic light buoys:
  - blue = start / end markers
  - red = do-not-go boundary
  - green = good passage boundary
- a static downward-facing 1080p camera.

## Conda (robotx) Python packages

Install Python-side packages used by dataset generation / CV scripts:

```bash
conda activate robotx
python -m pip install --upgrade pip
python -m pip install numpy opencv-python ultralytics pyyaml matplotlib
```

For ROS 2 bridge / Gazebo systems, use system ROS 2 Jazzy packages (apt), not conda.

## 1) Clone Base Repositories

```bash
mkdir -p ~/robotx_sim_ws/src
cd ~/robotx_sim_ws/src

# Water / wave simulation
git clone https://github.com/srmainwaring/asv_wave_sim.git

# Official VRX models / assets
git clone https://github.com/osrf/vrx.git
```

## 2) Install Dependencies (macOS, Homebrew)

```bash
brew update
brew tap osrf/simulation
brew install gz-harmonic cmake pkg-config ffmpeg
```

## 3) Build Beacon Plugin (macOS native)

```bash
cd /Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/plugins/robotx_beacon_plugin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

export GZ_SIM_SYSTEM_PLUGIN_PATH="$GZ_SIM_SYSTEM_PLUGIN_PATH:/Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/plugins/robotx_beacon_plugin/build"
```

## 4) Export Resource + Plugin Paths (macOS)

```bash
export GZ_SIM_RESOURCE_PATH="$HOME/robotx_sim_ws/src/asv_wave_sim/gz-waves-models/models:$HOME/robotx_sim_ws/src/asv_wave_sim/gz-waves-models/world_models:$HOME/robotx_sim_ws/src/asv_wave_sim/gz-waves-models/worlds:/Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/models:$HOME/robotx_sim_ws/src/vrx/vrx_gz/models"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/robotx_sim_ws/src/asv_wave_sim/gz-waves/build/lib:/Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/plugins/robotx_beacon_plugin/build"
export GZ_RENDERING_PLUGIN_PATH="$HOME/robotx_sim_ws/src/asv_wave_sim/gz-waves/build/lib"
```

## 5) Launch on macOS (server + GUI in separate terminals)

Terminal A:

```bash
/Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/scripts/run_robotx_world.sh
```

Terminal B:

```bash
/Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/scripts/run_robotx_gui.sh
```

If you see:

- `Failed to load plugin [gz-waves1-rendering-ogre2] : couldn't load library on path []`

then `GZ_RENDERING_PLUGIN_PATH` is missing in the terminal running `gz sim -g`.

To launch a specific generated domain-randomized world:

```bash
/Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/scripts/run_robotx_world.sh \
  /Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/worlds/generated/<scenario>.sdf
```

## 6) Optional Linux / ROS 2 Jazzy Dependencies

For ROS bridge / recorder flows (Ubuntu only):

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  python3-colcon-common-extensions \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-sim-vendor \
  ros-jazzy-gz-msgs-vendor \
  ros-jazzy-sdformat-vendor
```

## 7) Build the Custom Plugin Workspace (Ubuntu / Jazzy)

```bash
mkdir -p ~/robotx_sim_ws/src/robotx_custom
cd ~/robotx_sim_ws/src/robotx_custom

# Copy this repo's gazebo assets into your Gazebo workspace
cp -r /Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/* .

cd ~/robotx_sim_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 8) Export Resource + Plugin Paths (Ubuntu / Jazzy)

```bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:\
~/robotx_sim_ws/src/asv_wave_sim/gz-waves-models/models:\
~/robotx_sim_ws/src/asv_wave_sim/gz-waves-models/world_models:\
~/robotx_sim_ws/src/asv_wave_sim/gz-waves-models/worlds:\
~/robotx_sim_ws/src/robotx_custom/models:\
~/robotx_sim_ws/src/vrx/vrx_gz/models

export GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH:\
~/robotx_sim_ws/install/robotx_beacon_plugin/lib
```

## 9) Launch the World

```bash
gz sim -v4 ~/robotx_sim_ws/src/robotx_custom/worlds/robotx_task1_uav_view.sdf
```

Camera topic:

- `/robotx/uav/camera/image_raw`

## 10) Launch World + Bridge + Recorder Together

```bash
source /opt/ros/jazzy/setup.bash
source ~/robotx_sim_ws/install/setup.bash
ros2 launch ~/robotx_sim_ws/src/robotx_custom/launch/robotx_task1_uav.launch.py
```

Recorded frames are saved by default to:

- `captures/gazebo_uav/`

## 11) Files in this Folder

- `worlds/robotx_task1_uav_view.sdf`
  - custom world with harsh sun setup, asv wave plugin, buoy layout, and top-down camera.
- `models/robotx_buoy_red_led/model.sdf`
- `models/robotx_buoy_green_led/model.sdf`
- `models/robotx_buoy_blue_led/model.sdf`
  - modified buoy models with static top-visible semantic lights.
- `plugins/robotx_beacon_plugin/src/robotx_beacon_plugin.cpp`
  - optional beacon controller (legacy flashing modes) if you want time-varying lights
- `launch/robotx_task1_uav.launch.py`
  - launches Gazebo + `ros_gz_bridge` + recorder script
- `config/bridge_robotx_uav.yaml`
  - bridge mapping from Gazebo image topic to ROS 2 image topic
- `scripts/record_uav_dataset.py`
  - saves bridged ROS image frames to disk for dataset generation
- `scripts/generate_domain_randomized_worlds.py`
  - generates many `.sdf` world variants that sweep glare, wave model, and buoy layouts
  - writes `worlds/generated/manifest.json` with scenario metadata

## 12) Domain Randomization Sweep (glare / waves / buoy layout)

Generate scenario worlds:

```bash
python /Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/scripts/generate_domain_randomized_worlds.py
```

Interactive step-through mode (press Enter for next scenario):

```bash
python /Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/scripts/generate_domain_randomized_worlds.py --interactive
```

The script creates combinations of:

- glare profile: `mild`, `harsh`, `sunset_glint`
- waves model: `ocean_waves`, `regular_waves`, `trochoid_waves`
- buoy layout: `baseline`, `offset`, `wide_lane`
- semantic labels held constant across all cases:
  - blue = start/end
  - red = do-not-go
  - green = good passage

Then launch a generated scenario from `manifest.json`:

```bash
gz sim -s -r /Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/worlds/generated/<scenario>.sdf
# second terminal
gz sim -g
```

Current buoy models in this repo are configured for static top-visible lights by default.
