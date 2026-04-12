#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import zenoh
import json
from config import BRIDGE_CONFIG, ZENOH_CONNECT
from utils import get_msg_class, msg_to_dict, dict_to_msg

class CustomZenohBridgeROS2(Node):
    def __init__(self):
        super().__init__('custom_zenoh_bridge_ros2')
        
        # 建立 Zenoh 節點，主動連向 ROS 1 端的 TCP
        conf = zenoh.Config()
        conf.insert_json5("connect/endpoints", ZENOH_CONNECT)
        self.get_logger().info(f"[ROS2 Bridge] 初始化 Zenoh... connecting to {ZENOH_CONNECT}")
        self.session = zenoh.open(conf)
        
        # ==========================================
        # ROS 2 -> Zenoh (Pub_z)
        # ==========================================
        self.r2_subs = []
        for cfg in BRIDGE_CONFIG["ros2_to_ros1"]:
            topic = cfg["topic"]
            msg_class = get_msg_class(cfg["ros2_type"])
            zenoh_key = f"bridge/ros2_to_ros1{topic}"
            
            pub_z = self.session.declare_publisher(zenoh_key)
            
            # 使用 closure 來鎖定每個 topic 自己的 publisher
            def make_r2_callback(pz, t):
                def cb(msg):
                    payload = json.dumps(msg_to_dict(msg))
                    pz.put(payload)
                    # self.get_logger().info(f"[ROS2 -> Zenoh] Forwarded {t}: {payload}")
                return cb
                
            sub = self.create_subscription(msg_class, topic, make_r2_callback(pub_z, topic), 10)
            self.r2_subs.append(sub)
            self.get_logger().info(f"[ROS2 Bridge] 註冊轉發: {topic} ({cfg['ros2_type']}) -> Zenoh")

        # ==========================================
        # Zenoh -> ROS 2 (Sub_z)
        # ==========================================
        self.z_subs = []
        for cfg in BRIDGE_CONFIG["ros1_to_ros2"]:
            topic = cfg["topic"]
            msg_class = get_msg_class(cfg["ros2_type"])
            zenoh_key = f"bridge/ros1_to_ros2{topic}"
            
            pub_r2 = self.create_publisher(msg_class, topic, 10)
            
            # 使用 closure 來固定 publisher 與對應的 message 類別
            def make_z_callback(pr2, mc, t):
                def cb(sample):
                    try:
                        data = json.loads(sample.payload.to_string())
                        msg = mc()
                        msg = dict_to_msg(data, msg)
                        pr2.publish(msg)
                        # self.get_logger().info(f"[Zenoh -> ROS2] Received & Published {t}")
                    except Exception as e:
                        self.get_logger().error(f"JSON decode error on {t}: {e}")
                return cb
                
            sub = self.session.declare_subscriber(zenoh_key, make_z_callback(pub_r2, msg_class, topic))
            self.z_subs.append(sub)
            self.get_logger().info(f"[ROS2 Bridge] 註冊接收: Zenoh -> {topic} ({cfg['ros2_type']})")

        self.get_logger().info("[ROS2 Bridge] Custom Zenoh Bridge 啟動完成！")

def main(args=None):
    rclpy.init(args=args)
    node = CustomZenohBridgeROS2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
