# ==============================================================================
#  客製化 Zenoh Bridge 設定檔
#  在這裡新增你需要橋接的 Topic 與訊息類別
# ==============================================================================

BRIDGE_CONFIG = {
    # 方向： ROS 1 發佈，ROS 2 訂閱
    "ros1_to_ros2": [
        {
            "topic": "/bridge_test_r1_to_r2",
            "ros1_type": "std_msgs.msg.String",
            "ros2_type": "std_msgs.msg.String"
        },
        # 範例：車盤控制命令
        # {
        #     "topic": "/cmd_vel",
        #     "ros1_type": "geometry_msgs.msg.Twist",
        #     "ros2_type": "geometry_msgs.msg.Twist"
        # },
    ],
    
    # 方向： ROS 2 發佈，ROS 1 訂閱
    "ros2_to_ros1": [
        {
            "topic": "/bridge_test_r2_to_r1",
            "ros1_type": "std_msgs.msg.String",
            "ros2_type": "std_msgs.msg.String"
        },
        # 範例：里程計與坐標變換
        # {
        #     "topic": "/odom",
        #     "ros1_type": "nav_msgs.msg.Odometry",
        #     "ros2_type": "nav_msgs.msg.Odometry"
        # },
    ]
}

# 網路設定 (預設無需修改)
ZENOH_LISTEN = '["tcp/0.0.0.0:7447"]'
ZENOH_CONNECT = '["tcp/bridge:7447"]'
