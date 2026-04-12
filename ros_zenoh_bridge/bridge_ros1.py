#!/usr/bin/env python3
import rospy
import zenoh
import json
from config import BRIDGE_CONFIG, ZENOH_LISTEN
from utils import get_msg_class, msg_to_dict, dict_to_msg

def main():
    rospy.init_node('custom_zenoh_bridge_ros1', anonymous=True)
    
    # 建立 Zenoh 節點，並監聽對應 port 讓 ROS 2 端連線
    conf = zenoh.Config()
    conf.insert_json5("listen/endpoints", ZENOH_LISTEN)
    rospy.loginfo(f"[ROS1 Bridge] 初始化 Zenoh... listening on {ZENOH_LISTEN}")
    session = zenoh.open(conf)
    
    # ==========================================
    # ROS 1 -> Zenoh -> ROS 2 (Forwarding)
    # ==========================================
    for cfg in BRIDGE_CONFIG["ros1_to_ros2"]:
        topic = cfg["topic"]
        msg_class = get_msg_class(cfg["ros1_type"])
        zenoh_key = f"bridge/ros1_to_ros2{topic}"
        
        pub_z = session.declare_publisher(zenoh_key)
        
        # 使用 closure 固定 topic 對應的 publisher
        def make_r1_callback(pz, t):
            def cb(msg):
                payload = json.dumps(msg_to_dict(msg))
                pz.put(payload)
                # rospy.loginfo(f"[ROS1 -> Zenoh] Forwarded {t}: {payload}")
            return cb
            
        rospy.Subscriber(topic, msg_class, make_r1_callback(pub_z, topic))
        rospy.loginfo(f"[ROS1 Bridge] 註冊轉發: {topic} ({cfg['ros1_type']}) -> Zenoh")

    # ==========================================
    # ROS 2 -> Zenoh -> ROS 1 (Receiving)
    # ==========================================
    for cfg in BRIDGE_CONFIG["ros2_to_ros1"]:
        topic = cfg["topic"]
        msg_class = get_msg_class(cfg["ros1_type"])
        zenoh_key = f"bridge/ros2_to_ros1{topic}"
        
        pub_r1 = rospy.Publisher(topic, msg_class, queue_size=10)
        
        # 使用 closure 固定物件類別與對應 Publisher
        def make_z_callback(pr1, mc, t):
            def cb(sample):
                try:
                    data = json.loads(sample.payload.to_string())
                    msg = mc()
                    msg = dict_to_msg(data, msg)
                    pr1.publish(msg)
                    # rospy.loginfo(f"[Zenoh -> ROS1] Received & Published {t}")
                except Exception as e:
                    rospy.logerr(f"JSON decode error on {t}: {e}")
            return cb
                
        session.declare_subscriber(zenoh_key, make_z_callback(pub_r1, msg_class, topic))
        rospy.loginfo(f"[ROS1 Bridge] 註冊接收: Zenoh -> {topic} ({cfg['ros1_type']})")

    rospy.loginfo("[ROS1 Bridge] Custom Zenoh Bridge 啟動完成！")
    rospy.spin()

if __name__ == '__main__':
    main()
