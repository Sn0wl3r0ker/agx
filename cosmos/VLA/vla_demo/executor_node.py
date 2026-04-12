from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from .json_utils import dumps_json, loads_json
from .topics import CONTROL_COMMAND_TOPIC, ROBOT_STATUS_TOPIC


class VLAExecutorNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_executor")

        self.declare_parameter("linear_speed", 0.2)
        self.declare_parameter("approach_speed", 0.1)
        self.declare_parameter("angular_speed", 0.6)
        self.declare_parameter("turn_duration_sec", 2.1)
        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("status_hz", 2.0)

        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, ROBOT_STATUS_TOPIC, 10)
        self.create_subscription(
            String, CONTROL_COMMAND_TOPIC, self._on_control_command, 10
        )

        self.active_command_id = ""
        self.active_primitive = "stop_and_hold"
        self.active_step_id = None
        self.command_started_at = self._now()
        self.command_deadline: float | None = None
        self.turn_end_time: float | None = None
        self.executor_state = "IDLE"
        self.last_reason = "idle"

        control_period = 1.0 / float(self.get_parameter("control_hz").value)
        status_period = 1.0 / float(self.get_parameter("status_hz").value)
        self.create_timer(control_period, self._on_control_timer)
        self.create_timer(status_period, self._publish_status)

        self.get_logger().info(
            f"VLA executor ready. Listening on {CONTROL_COMMAND_TOPIC}, publishing /cmd_vel."
        )

    def _on_control_command(self, msg: String) -> None:
        try:
            payload = loads_json(msg.data, "control_command")
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        if not isinstance(payload, dict):
            self.get_logger().warn("Ignoring control command that is not a JSON object.")
            return
        command_id = str(payload.get("command_id", ""))
        if command_id and command_id == self.active_command_id:
            return
        self.active_command_id = command_id
        self.active_primitive = str(payload.get("control_primitive", "stop_and_hold"))
        self.active_step_id = payload.get("step_id")
        self.command_started_at = self._now()
        timeout_sec = payload.get("timeout_sec")
        self.command_deadline = (
            self.command_started_at + float(timeout_sec)
            if isinstance(timeout_sec, (int, float))
            else None
        )
        self.turn_end_time = None
        self.executor_state = "RUNNING"
        self.last_reason = "command_received"
        if self.active_primitive == "turn_right_90":
            self.turn_end_time = self.command_started_at + float(
                self.get_parameter("turn_duration_sec").value
            )
        elif self.active_primitive == "turn_left_90":
            self.turn_end_time = self.command_started_at + float(
                self.get_parameter("turn_duration_sec").value
            )
        elif self.active_primitive == "stop_and_hold":
            self.executor_state = "IDLE"
            self.last_reason = "hold_position"
            self._publish_twist(0.0, 0.0)
        self.get_logger().info(
            f"Executor command: {self.active_primitive} (step={self.active_step_id}, command_id={self.active_command_id})"
        )

    def _on_control_timer(self) -> None:
        now = self._now()
        if self.command_deadline is not None and now > self.command_deadline:
            self.executor_state = "ERROR"
            self.last_reason = "executor_timeout"
            self.active_primitive = "stop_and_hold"
            self._publish_twist(0.0, 0.0)
            return

        if self.active_primitive == "move_forward_until_recheck":
            self._publish_twist(float(self.get_parameter("linear_speed").value), 0.0)
            self.executor_state = "RUNNING"
        elif self.active_primitive == "approach_target_zone":
            self._publish_twist(float(self.get_parameter("approach_speed").value), 0.0)
            self.executor_state = "RUNNING"
        elif self.active_primitive == "turn_right_90":
            if self.turn_end_time is not None and now >= self.turn_end_time:
                self.active_primitive = "stop_and_hold"
                self.executor_state = "IDLE"
                self.last_reason = "turn_completed"
                self._publish_twist(0.0, 0.0)
            else:
                self._publish_twist(
                    0.0, -float(self.get_parameter("angular_speed").value)
                )
        elif self.active_primitive == "turn_left_90":
            if self.turn_end_time is not None and now >= self.turn_end_time:
                self.active_primitive = "stop_and_hold"
                self.executor_state = "IDLE"
                self.last_reason = "turn_completed"
                self._publish_twist(0.0, 0.0)
            else:
                self._publish_twist(
                    0.0, float(self.get_parameter("angular_speed").value)
                )
        else:
            self.active_primitive = "stop_and_hold"
            self.executor_state = "IDLE"
            self._publish_twist(0.0, 0.0)

    def _publish_twist(self, linear_x: float, angular_z: float) -> None:
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.vel_pub.publish(twist)

    def _publish_status(self) -> None:
        payload = {
            "executor_state": self.executor_state,
            "active_command_id": self.active_command_id,
            "active_primitive": self.active_primitive,
            "active_step_id": self.active_step_id,
            "reason": self.last_reason,
            "stamp": self._now(),
        }
        msg = String()
        msg.data = dumps_json(payload)
        self.status_pub.publish(msg)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLAExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_twist(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
