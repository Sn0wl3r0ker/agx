import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'slam_params_file': os.path.join(
                get_package_share_directory('car_control'),
                'config',
                'mapper_params_online_async.yaml'
            )
        }.items()
    )

    return LaunchDescription([
        slam_toolbox_launch
    ])