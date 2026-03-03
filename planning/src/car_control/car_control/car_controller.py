'''
需到 setup.py 添加 :
    setup 
        entry_points 的 console_scripts 下 添加來定義節點 : 
            'car_controller_node_2 = car_control.car_controller_2:main',  
            
        install_requires 下 添加引入的python套件
            ['setuptools','pyserial', 'tf-transformations'],

需到 package.xml 添加引用的ros2套件:
    <exec_depend>geometry_msgs</exec_depend>
    <exec_depend>nav_msgs</exec_depend>
    <exec_depend>tf2_ros</exec_depend>
    <exec_depend>tf2_msgs</exec_depend>
    <!-- 若需要 -->
    <exec_depend>std_msgs</exec_depend>

啟動改節點命令 : 
    ros2 run car_control car_controller_node_2
'''
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import serial
import struct
import time
import math
class car:
    def __init__(self):
        pass
    # 車控_工作區
    def car_init(self , car_distance = 0.25 , Tire_diameter = 0.092 , encoder_resolution = 11 , gear_ratio = 90, usb_port = '/dev/car'):
        self.drive_init(usb_port)   # 初始化驅動器
        # 車體參數
        #   車輪直徑 = 0.092 單位:米
        #   (現實 / 虛擬   )線性距離 * 目前輪距
        #   車輪距離 = 前後兩個輪胎中心點的距離 + 左右單位兩個輪胎中心點的距離 單位:米
        #       車子轉過頭爲數值調小 反之
        #   (虛擬 / 現實  )角度 * 目前car_distance
        self.car_distance = car_distance
        # --------
        # 馬達參數
        #   每圈的脈衝數 = 減速比 * 編碼器線數
        Total_pulses = gear_ratio * encoder_resolution
        # -------

        # 車輪半徑 = 車輪直徑 / 2 單位:米
        self.Tire_radius = Tire_diameter / 2 

        # 速度1每秒的速度 = (pi * 車輪直徑 / 一圈的脈衝數) * 速度1每秒的脈衝數
        self.speed_1_every_second_speed = (math.pi * Tire_diameter / Total_pulses) / 10
    def car_diff_wheel_set_speed(self, linear_x, angular_z):    # 差速輪速度換算並執行 速度(線性速度,轉速度)
        # 差速輪的速度計算
        left_speed = linear_x - (self.car_distance / 2) * angular_z
        right_speed = linear_x + (self.car_distance / 2) * angular_z
        #for i in range(2):
            #print(f"Wheel {i + 1} speed: {speed_list[i]}")
        # ---------------
        # 寫入驅動器
        self.set_drive_speed(0x01, int(left_speed / self.speed_1_every_second_speed ))
        self.set_drive_speed(0x02, int(right_speed / self.speed_1_every_second_speed ))
        #print(f"Left wheel speed: {int(left_speed / self.speed_1_every_second_speed )}, Right wheel speed: {int(right_speed / self.speed_1_every_second_speed )}")
        #print("-----------------------------------")
        # ---------------
    def car_mecanum_wheel_set_speed(self,linear_x, linear_y, angular_z):    # 麥克納輪速度換算並執行 速度(線性速度_x,線性速度_y,轉速度)
        # 麥克納輪的速度計算
        speed_list = [0, 0, 0, 0]
        speed_list[0] = ( linear_x - linear_y - (self.car_distance) * angular_z )
        speed_list[1] = ( linear_x + linear_y + (self.car_distance) * angular_z )
        speed_list[2] = ( linear_x + linear_y - (self.car_distance) * angular_z )
        speed_list[3] = ( linear_x - linear_y + (self.car_distance) * angular_z )
        #for i in range(4):
            #print(f"Wheel {i + 1} speed: {speed_list[i]}")
        # ---------------
        # 寫入驅動器
        for i in range(4):
            self.set_drive_speed(i + 1, int(speed_list[i] / self.speed_1_every_second_speed ))
            #print(f"Wheel {i + 1} speed: {int(speed_list[i] / self.speed_1_every_second_speed )}")
        #print("-----------------------------------")
        # ---------------
    # ---------
    # 驅動器_工作區
    def drive_init(self,usb_port):   # 驅動器初始化
        self.Drive = serial.Serial(
            port=usb_port,  # USB port for the car controller    
            baudrate=9600,
            # 8n2
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_TWO,
            timeout=1
        )
    def set_drive_speed(self, slave_id  , speed):   # 設定驅動器速度 (驅動器ID, 執行的速度)
        function_code = 0x06      # 寫單一暫存器
        register_address = 0x0043 # 暫存器地址
        # 轉成 2 bytes 大端格式
        value_bytes = struct.pack('>h', speed)
        # 組合封包（不含 CRC）
        frame = struct.pack('>B B H', slave_id, function_code, register_address)
        frame += value_bytes
        # 加上 CRC
        full_packet = frame + self.calc_crc(frame)
        #print(f"Sending: {full_packet.hex()}")
        self.Drive.write(full_packet)
        # 接收回應（寫單一暫存器通常回傳同樣的資料）
        response = self.Drive.read(8)
        #print(f"Received: {response.hex()}")
        # '''
    def calc_crc(self, data: bytes) -> bytes:   # 計算 CRC-16
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return struct.pack('<H', crc)  # CRC 是 little-endian
    # ------------

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf_transformations
import tf2_ros
from geometry_msgs.msg import TransformStamped

class ros2_car(Node):
    def __init__(self):
        super().__init__('ros2_car_node')   # 初始化(節點名稱)

        # ros訊息初始化
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.odom_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.odom_up_hz = 10 # odom 更新頻率
        # -----------

        # ros_odom訊息資料暫存
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0
        self.last_time = self.get_clock().now().to_msg()
        self.if_set_speed = True
        # -------------------

        # 車控初始化
        #   ros車子參數讀取
        self.declare_parameter('car_distance', 0.25)
        self.declare_parameter('Tire_diameter', 0.092)
        self.declare_parameter('encoder_resolution', 11)
        self.declare_parameter('gear_ratio', 90)
        self.declare_parameter('odom_up_hz', 10)
        self.declare_parameter('car_mode', 'diff')  # 'car_diff_wheel' or 'car_mecanum_wheel'
        self.declare_parameter('usb_port', '/dev/car')  # USB port for the car controller
        #  -------------

        #   參數放入
        car_distance = self.get_parameter('car_distance').value
        Tire_diameter = self.get_parameter('Tire_diameter').value
        encoder_resolution = self.get_parameter('encoder_resolution').value
        gear_ratio = self.get_parameter('gear_ratio').value
        usb_port = self.get_parameter('usb_port').value
        self.odom_up_hz = self.get_parameter('odom_up_hz').value
        self.car_mode = self.get_parameter('car_mode').value
        #   -------
        
        #   顯示參數
        self.get_logger().info(f'car_distance: {car_distance}')
        self.get_logger().info(f'Tire_diameter: {Tire_diameter}')
        self.get_logger().info(f'encoder_resolution: {encoder_resolution}')
        self.get_logger().info(f'gear_ratio: {gear_ratio}')
        self.get_logger().info(f'usb_port: {usb_port}')
        self.get_logger().info(f'odom_up_hz: {self.odom_up_hz}')
        self.get_logger().info(f'car_mode: {self.car_mode}')
        #   -------

        self.car = car()
        self.car.car_init(car_distance = car_distance , Tire_diameter = Tire_diameter , encoder_resolution = encoder_resolution , gear_ratio = gear_ratio, usb_port = usb_port)  # 初始化車控參數
        # --------

    def cmd_vel_callback(self, msg):    # 訂閱要執行的速度訊息
        # 取得速度
        self.linear_x = msg.linear.x
        self.linear_y = msg.linear.y
        self.angular_z = msg.angular.z
        self.if_set_speed = False
        #print(f"linear_x: {self.linear_x}, linear_y: {self.linear_y}, angular_z: {self.angular_z}")
    def ros2_car_exe(self):
        rate = self.create_rate(self.odom_up_hz)
        while rclpy.ok():
            current_time = self.get_clock().now().to_msg()
            # 計算odom
            dt = (current_time.sec + current_time.nanosec * 1e-9) - (self.last_time.sec + self.last_time.nanosec * 1e-9)
            delta_x = (self.linear_x * math.cos(self.theta) - self.linear_y * math.sin(self.theta)) * dt
            delta_y = (self.linear_x * math.sin(self.theta) + self.linear_y * math.cos(self.theta)) * dt
            delta_th = self.angular_z * dt
            self.x += delta_x
            self.y += delta_y
            self.theta += delta_th
            #print(f"x: {self.x}, y: {self.y}, theta: {self.theta} dt: {dt}")
            # ---------
            
            # 發布 TF
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            q = tf_transformations.quaternion_from_euler(0, 0, self.theta)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.odom_broadcaster.sendTransform(t)

            # 發布 Odometry
            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation.x = q[0]
            odom.pose.pose.orientation.y = q[1]
            odom.pose.pose.orientation.z = q[2]
            odom.pose.pose.orientation.w = q[3]
            odom.twist.twist.linear.x = self.linear_x
            odom.twist.twist.linear.y = self.linear_y
            odom.twist.twist.angular.z = self.angular_z
            self.odom_pub.publish(odom)
            self.last_time = current_time
            
            # 設定車輛速度
            if self.if_set_speed == False : 
                if self.car_mode == 'mecanum':
                    # 麥克納輪
                    self.car.car_mecanum_wheel_set_speed(self.linear_x, self.linear_y, self.angular_z)
                elif self.car_mode == 'diff':
                    # 差速輪
                    self.car.car_diff_wheel_set_speed(self.linear_x, self.angular_z)
            self.if_set_speed = True
            rclpy.spin_once(self) 

def main(args=None):
    rclpy.init(args=args)
    node = ros2_car()
    try:
        node.ros2_car_exe()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
        