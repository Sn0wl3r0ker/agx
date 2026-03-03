import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

class TFListenerNode(Node):
    def __init__(self):
        super().__init__('tf_listener_node')
        
        # 初始化 TF2 Buffer 和 TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 定時器，用於定期查詢 TF
        self.timer = self.create_timer(1.0, self.lookup_transform)  # 每秒查詢一次
        self.get_logger().info('TF Listener Node Initialized')

    def lookup_transform(self):
        try:
            # 查詢 map 到 base_link 的座標轉換
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            self.get_logger().info(
                f'Transform from map to base_link: '
                f'translation=({transform.transform.translation.x}, '
                f'{transform.transform.translation.y}, '
                f'{transform.transform.translation.z}), '
                f'rotation=({transform.transform.rotation.x}, '
                f'{transform.transform.rotation.y}, '
                f'{transform.transform.rotation.z}, '
                f'{transform.transform.rotation.w})'
            )
        except Exception as e:
            self.get_logger().warn(f'Failed to lookup transform: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TFListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()