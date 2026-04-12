from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="vla_demo",
                executable="vla_executor",
                name="vla_executor",
                output="screen",
            ),
        ]
    )
