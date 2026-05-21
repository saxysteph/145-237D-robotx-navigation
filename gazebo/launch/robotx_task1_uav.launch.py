from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_file = Path(__file__).resolve()
    gazebo_root = launch_file.parent.parent
    world_default = gazebo_root / "worlds" / "robotx_task1_uav_view.sdf"
    bridge_yaml = gazebo_root / "config" / "bridge_robotx_uav.yaml"
    recorder_script = gazebo_root / "scripts" / "record_uav_dataset.py"

    world_arg = DeclareLaunchArgument(
        "world",
        default_value=str(world_default),
        description="Path to Gazebo world file",
    )
    out_dir_arg = DeclareLaunchArgument(
        "output_dir",
        default_value=str(gazebo_root.parent / "captures" / "gazebo_uav"),
        description="Directory for saved camera frames",
    )

    # Match gazebo/README.md + gz_env.sh: waves/VRX live under ~/robotx_sim_ws/src (not robotx_gz_ws).
    wave_root = Path.home() / "robotx_sim_ws" / "src" / "asv_wave_sim" / "gz-waves-models"
    resource_path = ":".join(
        [
            str(gazebo_root / "models"),
            str(wave_root / "models"),
            str(wave_root / "world_models"),
            str(wave_root / "worlds"),
            str(Path.home() / "robotx_sim_ws" / "src" / "vrx" / "vrx_gz" / "models"),
        ]
    )
    set_resource_env = SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path)

    gz = ExecuteProcess(
        cmd=["gz", "sim", "-r", LaunchConfiguration("world")],
        output="screen",
    )
    bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "--ros-args",
            "-p",
            f"config_file:={bridge_yaml}",
        ],
        output="screen",
    )
    recorder = ExecuteProcess(
        cmd=[
            "python3",
            str(recorder_script),
            "--topic",
            "/robotx/uav/camera/image_raw",
            "--out-dir",
            LaunchConfiguration("output_dir"),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            world_arg,
            out_dir_arg,
            set_resource_env,
            gz,
            TimerAction(period=2.5, actions=[bridge]),
            TimerAction(period=4.0, actions=[recorder]),
        ]
    )
