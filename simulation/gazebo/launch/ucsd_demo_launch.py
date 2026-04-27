from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_file = Path(__file__).resolve()
    launch_dir = launch_file.parent
    gazebo_root = launch_dir.parent

    default_world = gazebo_root / "worlds" / "ucsd_robotx_demo.world.sdf"
    drone_model = gazebo_root / "models" / "ucsd_drone" / "model.sdf"

    # Support both repository layout and vrx_gz package layout.
    bridge_candidates = [
        gazebo_root.parent / "ros2_bridge" / "bridge.yaml",  # repo/simulation layout
        gazebo_root / "config" / "bridge.yaml",  # vrx_gz layout
    ]
    bridge_config = next((p for p in bridge_candidates if p.exists()), bridge_candidates[0])

    world_arg = DeclareLaunchArgument(
        "world",
        default_value=str(default_world),
        description="Absolute path to SDF world file.",
    )

    # Include this repository's Gazebo model paths and VRX paths if present.
    resource_path = ":".join(
        [
            str(gazebo_root),
            str(gazebo_root / "models"),
            str(Path.home() / "vrx_ws" / "src" / "vrx" / "vrx_gz"),
            str(Path.home() / "vrx_ws" / "src" / "vrx" / "vrx_urdf"),
        ]
    )

    set_resource_env = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=resource_path,
    )

    gz = ExecuteProcess(
        cmd=["gz", "sim", "-r", LaunchConfiguration("world")],
        output="screen",
    )

    spawn_drone = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-name",
            "ucsd_drone",
            "-file",
            str(drone_model),
            "-x",
            "60",
            "-y",
            "2",
            "-z",
            "25",
        ],
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
            f"config_file:={bridge_config}",
        ],
        output="screen",
    )

    topic_list = TimerAction(
        period=8.0,
        actions=[ExecuteProcess(cmd=["ros2", "topic", "list"], output="screen")],
    )

    return LaunchDescription(
        [
            world_arg,
            set_resource_env,
            gz,
            TimerAction(period=2.0, actions=[spawn_drone]),
            TimerAction(period=3.0, actions=[bridge]),
            topic_list,
        ]
    )
