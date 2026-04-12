from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mission_file = LaunchConfiguration("mission_file")
    autostart = LaunchConfiguration("autostart")
    camera_device = LaunchConfiguration("camera_device")

    return LaunchDescription(
        [
            DeclareLaunchArgument("mission_file", default_value=""),
            DeclareLaunchArgument("autostart", default_value="false"),
            DeclareLaunchArgument("camera_device", default_value="/dev/video0"),
            Node(
                package="vla_demo",
                executable="camera_front_node",
                name="camera_front_node",
                output="screen",
                parameters=[{"video_device": camera_device}],
            ),
            Node(
                package="vla_demo",
                executable="vla_mission_manager",
                name="vla_mission_manager",
                output="screen",
                parameters=[
                    {
                        "default_mission_file": mission_file,
                        "autostart": autostart,
                    }
                ],
            ),
        ]
    )
