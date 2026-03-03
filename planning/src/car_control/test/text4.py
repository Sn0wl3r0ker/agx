import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class FixedPointNavigator(Node):
    def __init__(self):
        super().__init__('fixed_point_navigator')
        
        # 固定的導航座標點
        self.points = [
            {'x': 5.36, 'y': -0.34, 'frame_id': 'map'},
            {'x': -1.0, 'y': 0.0, 'frame_id': 'map'},
            {'x': -1.08, 'y': 2.01, 'frame_id': 'map'},
            {'x': -1.0, 'y': 0.0, 'frame_id': 'map'},
        ]
        self.current_index = 0
        
        # 發佈導航座標
        self.navigation_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # 定時器，用於循環發佈座標
        self.timer = self.create_timer(20.0, self.publish_next_point)  # 每 5 秒發佈一次座標
        self.get_logger().info('Fixed Point Navigator Initialized')

    def publish_next_point(self):
        # 獲取當前座標點
        point = self.points[self.current_index]
        
        # 構建 PoseStamped 消息
        navigation_msg = PoseStamped()
        navigation_msg.header.stamp = self.get_clock().now().to_msg()
        navigation_msg.header.frame_id = point['frame_id']
        navigation_msg.pose.position.x = point['x']
        navigation_msg.pose.position.y = point['y']
        navigation_msg.pose.orientation.w = 1.0  # 假設方向固定
        
        # 發佈導航座標
        self.navigation_publisher.publish(navigation_msg)
        self.get_logger().info(f'Published navigation pose: x={point["x"]}, y={point["y"]}')
        
        # 更新索引以循環座標點
        self.current_index = (self.current_index + 1) % len(self.points)

def main(args=None):
    rclpy.init(args=args)
    node = FixedPointNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()