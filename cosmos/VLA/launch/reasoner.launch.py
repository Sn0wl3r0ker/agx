from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    backend_mode = LaunchConfiguration("backend_mode")

    return LaunchDescription(
        [
            DeclareLaunchArgument("backend_mode", default_value="mock"),
            Node(
                package="vla_demo",
                executable="cosmos_reasoner_node",
                name="cosmos_reasoner_node",
                output="screen",
                parameters=[{"backend_mode": backend_mode}],
            ),
        ]
    )
