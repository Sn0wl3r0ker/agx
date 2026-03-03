'''
啟動 車控 與 lidar 節點 並發布 tf
啟動命令
    ros2 launch car_control car_sensor.launch.py
'''

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    

    # 車控_工作區_開始
    #   取得 car_control 的參數檔
    #       到 (car_control) 專案的路徑,
    #       的 (config) 目錄下,
    #       取得 (car_controller.yaml) 參數檔案
    car_controller_config = os.path.join(
        get_package_share_directory('car_control'),
        'config',
        'car_controller.yaml'
    )
    
    #   宣告一個車控節點
    #       執行內容為(car_control)專案的
    #       的(car_controller_node_2.py)節點
    #       並設定名稱為(car_controller_node)
    #       設定輸出到終端機
    #       設定模擬終端機輸出
    #       並配置(car_controller_config)參數檔到節點
    car_controller_node = Node(
        package='car_control',
        executable='car_controller_node',
        name='car_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[car_controller_config]
    )
    # 車控_工作區_結束

    # return 放入要執行的節點
    return LaunchDescription([
        car_controller_node,
    ])