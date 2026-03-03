import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import time
class MoveRobot(Node):
    def __init__(self):
        super().__init__('move_robot')
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.current_position = None
        self.current_orientation = None

    def odom_callback(self, msg):
        # 更新當前位置和方向
        self.current_position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.current_orientation = yaw

    def move(self, A_few_laps, speed):
        twist = Twist()
        twist.angular.z = speed
        time.sleep(3)
        self.cmd_vel_publisher.publish(twist)
        start_position = None
        if_A_few_laps = True
        A_few_laps_count = 0
        while rclpy.ok():
            
            rclpy.spin_once(self)
            if self.current_orientation is None:
                continue


            # 計算移動距離
            print("current_orientation", self.current_orientation)
            if self.current_orientation < 0.0 and if_A_few_laps:
                if_A_few_laps = False
            elif self.current_orientation > 0.0 and not if_A_few_laps:
                if_A_few_laps = True
                A_few_laps_count += 1

            if  A_few_laps_count >= A_few_laps:
                break

        # 停止機器人
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

        # 顯示目前 odom 資訊
        print(self.current_orientation * 57.32)
        return self.current_orientation * 57.32

def main(args=None):
    rclpy.init(args=args)
    node = MoveRobot()

    try:
        speed = 1.0
        odom_up_hz = 10
        a = node.move(1.0, speed) # 移動 1 公尺，速度 0.3 m/s
        a_Reality = float(input("輸入實際的車頭角: "))
        a_error_1 = a_Reality - (360 + a)

        print(f"誤差: {a_error_1:.2f} 度")

        a = node.move(2.0, speed) # 移動 1 公尺，速度 0.3 m/s
        a_Reality = float(input("輸入實際的車頭角: "))
        a_error_2 = a_Reality - (360 + a)

        print(f"誤差: {a_error_2:.2f} 度")
        a_Reality_error = a_error_2 - a_error_1 

        print(f"固定誤差: {a_Reality_error:.2f} 度")

        car_distance = 360 / (360 + a_Reality_error)  * float(input("目前車距: "))
        print(f"換算後車距: {car_distance:.2f} m")

        acc_error = (a_error_1 + a_error_2) / 2
        print(f"acc_error: {acc_error:.2f} m")

        acc_angle_speed = acc_error / speed / odom_up_hz * 0.017

        print(f"acc_angle_speed: {acc_angle_speed:.2f} m/s")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()