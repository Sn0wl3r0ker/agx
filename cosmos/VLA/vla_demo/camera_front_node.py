from __future__ import annotations

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .topics import CAMERA_IMAGE_TOPIC


class CameraFrontNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_front_node")

        self.declare_parameter("video_device", "/dev/video0")
        self.declare_parameter("capture_hz", 10.0)
        self.declare_parameter("frame_id", "camera_front_optical_frame")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("backend", "v4l2")

        self.publisher = self.create_publisher(Image, CAMERA_IMAGE_TOPIC, 10)
        self.bridge = CvBridge()
        self.capture = None
        self.last_warn = 0.0

        period = 1.0 / float(self.get_parameter("capture_hz").value)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"Camera front node ready. Publishing {CAMERA_IMAGE_TOPIC} from "
            f"{self.get_parameter('video_device').value}."
        )

    def _open_capture(self) -> None:
        device = str(self.get_parameter("video_device").value)
        backend_name = str(self.get_parameter("backend").value).lower()
        backend = cv2.CAP_V4L2 if backend_name == "v4l2" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(device, backend)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.get_parameter("width").value))
        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT, float(self.get_parameter("height").value)
        )

    def _on_timer(self) -> None:
        if self.capture is None or not self.capture.isOpened():
            self._open_capture()
            if self.capture is None or not self.capture.isOpened():
                self._throttled_warn("Failed to open front camera.")
                return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self._throttled_warn("Camera read failed.")
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("frame_id").value)
        self.publisher.publish(msg)

    def _throttled_warn(self, text: str) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_warn > 2.0:
            self.get_logger().warn(text)
            self.last_warn = now


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraFrontNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.capture is not None:
            node.capture.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
