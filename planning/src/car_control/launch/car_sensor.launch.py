'''
統一感測器啟動檔 — 車控節點 + LiDAR
啟動命令：
    ros2 launch car_control car_sensor.launch.py

架構說明：
    此 Launch 檔整合所有感測器相關的 Node，取代過去需要分別啟動
    car_control 和 urg_node2 的流程。

    第三方套件 (urg_node2) 保持唯讀，其參數透過
    car_control/config/urg_node2_override.yaml 注入覆寫，
    遵循 ROS 2 的 Open-Closed Principle。

Data Flow (下行)：
    /cmd_vel → [kinematics_node] → /motor_cmd → [serial_bridge_node] → UART TX {"ls":x,"rs":y}
    /charge_cmd → [serial_bridge_node] → UART TX {"charge":1}

Data Flow (上行)：
    UART RX {"p1":x,"p2":y}          → [serial_bridge_node] → /raw_odom
    UART RX {"pow":24.5}             → [serial_bridge_node] → /battery_state  → [kinematics_node] → /battery_voltage
    UART RX {"can_v":x,"can_w":y...} → [serial_bridge_node] → /charge_status → [kinematics_node] → /charging_state

狀態推估 (Sensor Fusion)：
    /raw_odom + /camera/imu → [ekf_node] → /odometry/filtered + odom->base_footprint TF

LiDAR：
    urg_node2 (LifecycleNode) → /scan (sensor_msgs/LaserScan)
'''

import os
import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():

    # ==========================================================
    #  參數檔路徑
    # ==========================================================

    # car_control 自有節點的參數
    car_control_config = os.path.join(
        get_package_share_directory('car_control'),
        'config',
        'car_controller_cpp.yaml',
    )

    # EKF 參數檔
    ekf_config = os.path.join(
        get_package_share_directory('car_control'),
        'config',
        'ekf.yaml',
    )

    # urg_node2 覆寫參數 — 放在 car_control 內，第三方套件保持唯讀
    urg_override_config = os.path.join(
        get_package_share_directory('car_control'),
        'config',
        'urg_node2_override.yaml',
    )

    # URDF XACRO 檔案路徑
    xacro_file = os.path.join(
        get_package_share_directory('car_control'),
        'urdf',
        'amr_core.urdf.xacro'
    )
    
    # 讀取 xacro 並轉譯成純 URDF xml 字串
    doc = xacro.process_file(xacro_file)
    robot_description = {'robot_description': doc.toxml()}

    # 讀取 YAML 並取得 ros__parameters dict
    with open(urg_override_config, 'r') as f:
        urg_params = yaml.safe_load(f)['urg_node2']['ros__parameters']

    # ==========================================================
    #  Launch Arguments
    # ==========================================================
    auto_start_arg = DeclareLaunchArgument(
        'auto_start', default_value='true',
        description='自動將 urg_node2 LifecycleNode 轉換到 Active 狀態'
    )

    # ==========================================================
    #  urg_node2 — LifecycleNode
    # ==========================================================
    urg_lifecycle_node = LifecycleNode(
        package='urg_node2',
        executable='urg_node2_node',
        name='urg_node2',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=[urg_params],           # ← 注入我們的覆寫參數
        remappings=[('scan', 'scan')],
    )

    # Unconfigured → Inactive (configure)
    configure_handler = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=urg_lifecycle_node,
            on_start=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(urg_lifecycle_node),
                        transition_id=Transition.TRANSITION_CONFIGURE,
                    ),
                ),
            ],
        ),
        condition=IfCondition(LaunchConfiguration('auto_start')),
    )

    # Inactive → Active (activate)
    activate_handler = RegisterEventHandler(
        event_handler=OnStateTransition(
            target_lifecycle_node=urg_lifecycle_node,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(urg_lifecycle_node),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    ),
                ),
            ],
        ),
        condition=IfCondition(LaunchConfiguration('auto_start')),
    )

    # ==========================================================
    #  car_control — C++ 車控節點
    # ==========================================================
    serial_bridge_node = Node(
        package='car_control',
        executable='serial_bridge_node',
        name='serial_bridge_node',
        output='screen',
        emulate_tty=True,
        parameters=[car_control_config],
    )

    kinematics_node = Node(
        package='car_control',
        executable='kinematics_node',
        name='kinematics_node',
        output='screen',
        emulate_tty=True,
        parameters=[car_control_config],
    )

    # ==========================================================
    #  EKF (Robot Localization)
    # ==========================================================
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config],
    )

    # ==========================================================
    #  Robot State Publisher (處理 URDF 靜態轉換)
    # ==========================================================
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': False}]
    )


    # ==========================================================
    #  組合啟動
    # ==========================================================
    return LaunchDescription([
        # Launch Arguments
        auto_start_arg,

        # LiDAR (LifecycleNode + 自動狀態轉換)
        urg_lifecycle_node,
        configure_handler,
        activate_handler,

        # 車控節點
        serial_bridge_node,
        kinematics_node,

        # EKF 濾波器 (融合里程計與 IMU)
        ekf_node,

        # URDF / TF (所有靜態座標樹由 robot_state_publisher 統一發布)
        rsp_node,
    ])

