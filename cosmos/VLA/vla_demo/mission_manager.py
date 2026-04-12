from __future__ import annotations

from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .json_utils import dumps_json, loads_json
from .mission_loader import default_mission_path, load_mission_file
from .schemas import MissionSpec, SchemaError
from .state_machine import MissionStateMachine
from .topics import (
    ABORT_TOPIC,
    CONTROL_COMMAND_TOPIC,
    CURRENT_PROMPT_TOPIC,
    CURRENT_STEP_TOPIC,
    INFERENCE_INTERVAL_TOPIC,
    INFERENCE_RESULT_TOPIC,
    MISSION_STATE_TOPIC,
    ROBOT_STATUS_TOPIC,
    SET_MISSION_TOPIC,
)


class VLAMissionManager(Node):
    def __init__(self) -> None:
        super().__init__("vla_mission_manager")

        self.declare_parameter("default_mission_file", str(default_mission_path()))
        self.declare_parameter("autostart", False)
        self.declare_parameter("publish_period_sec", 0.5)
        self.declare_parameter("robot_status_timeout_sec", 2.5)
        self.declare_parameter("inference_stale_multiplier", 3.0)

        self.machine = MissionStateMachine(
            robot_status_timeout_sec=float(
                self.get_parameter("robot_status_timeout_sec").value
            ),
            inference_stale_multiplier=float(
                self.get_parameter("inference_stale_multiplier").value
            ),
        )

        self.current_step_pub = self.create_publisher(String, CURRENT_STEP_TOPIC, 10)
        self.current_prompt_pub = self.create_publisher(String, CURRENT_PROMPT_TOPIC, 10)
        self.inference_interval_pub = self.create_publisher(
            String, INFERENCE_INTERVAL_TOPIC, 10
        )
        self.control_command_pub = self.create_publisher(
            String, CONTROL_COMMAND_TOPIC, 10
        )
        self.mission_state_pub = self.create_publisher(String, MISSION_STATE_TOPIC, 10)

        self.create_subscription(String, SET_MISSION_TOPIC, self._on_set_mission, 10)
        self.create_subscription(
            String, INFERENCE_RESULT_TOPIC, self._on_inference_result, 10
        )
        self.create_subscription(String, ROBOT_STATUS_TOPIC, self._on_robot_status, 10)
        self.create_subscription(String, ABORT_TOPIC, self._on_abort, 10)

        self._last_published: dict[str, str] = {}
        self.timer = self.create_timer(
            float(self.get_parameter("publish_period_sec").value), self._on_timer
        )

        autostart = bool(self.get_parameter("autostart").value)
        if autostart:
            mission_file = str(self.get_parameter("default_mission_file").value).strip()
            self._start_mission_from_path(mission_file)

        self.get_logger().info(
            "VLA mission manager ready. Topics: "
            f"{SET_MISSION_TOPIC}, {CURRENT_PROMPT_TOPIC}, {CONTROL_COMMAND_TOPIC}, {MISSION_STATE_TOPIC}"
        )

    def _on_set_mission(self, msg: String) -> None:
        raw = msg.data.strip()
        if not raw:
            self.get_logger().warn("Received empty mission request.")
            return
        if raw.startswith("{"):
            payload = loads_json(raw, "set_mission")
            if isinstance(payload, dict) and "mission_file" in payload:
                mission_file = str(payload["mission_file"])
                self._start_mission_from_path(mission_file)
                return
            try:
                mission = MissionSpec.from_dict(payload)
            except SchemaError as exc:
                self.get_logger().error(f"Mission schema rejected: {exc}")
                return
            self.machine.start_mission(mission, self._now())
            self.get_logger().info(f"Mission started from inline payload: {mission.mission_id}")
            self._publish_all(force=True)
            return
        self._start_mission_from_path(raw)

    def _start_mission_from_path(self, mission_file: str) -> None:
        try:
            mission = load_mission_file(mission_file)
        except (OSError, ValueError, SchemaError) as exc:
            self.get_logger().error(f"Failed to load mission '{mission_file}': {exc}")
            return
        self.machine.start_mission(mission, self._now())
        self.get_logger().info(f"Mission started: {mission.mission_id} ({mission_file})")
        self._publish_all(force=True)

    def _on_inference_result(self, msg: String) -> None:
        try:
            payload = loads_json(msg.data, "inference_result")
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        if not isinstance(payload, dict):
            self.get_logger().warn("Ignoring inference result that is not a JSON object.")
            return
        self.machine.handle_inference(payload, self._now())
        self._publish_all(force=False)

    def _on_robot_status(self, msg: String) -> None:
        try:
            payload = loads_json(msg.data, "robot_status")
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        if not isinstance(payload, dict):
            self.get_logger().warn("Ignoring robot status that is not a JSON object.")
            return
        self.machine.handle_robot_status(payload, self._now())

    def _on_abort(self, msg: String) -> None:
        reason = msg.data.strip() or "operator_abort"
        self.machine.abort(reason, self._now())
        self.get_logger().warn(f"Mission aborted: {reason}")
        self._publish_all(force=True)

    def _on_timer(self) -> None:
        self.machine.tick(self._now())
        self._publish_all(force=False)

    def _publish_all(self, force: bool) -> None:
        if self.machine.has_active_mission:
            self._publish_if_changed(
                "current_step",
                self.current_step_pub,
                self.machine.build_current_step_payload(),
                force,
            )
            self._publish_if_changed(
                "current_prompt",
                self.current_prompt_pub,
                self.machine.build_prompt_payload(),
                force,
            )
            self._publish_if_changed(
                "inference_interval",
                self.inference_interval_pub,
                self.machine.build_inference_interval_payload(),
                force,
            )
            self._publish_if_changed(
                "control_command",
                self.control_command_pub,
                self.machine.build_control_command_payload(),
                force,
            )
        state_payload = self.machine.build_mission_state_payload(self._now())
        msg = String()
        msg.data = dumps_json(state_payload)
        self.mission_state_pub.publish(msg)

    def _publish_if_changed(
        self,
        key: str,
        publisher,
        payload: dict,
        force: bool,
    ) -> None:
        raw = dumps_json(payload)
        if not force and self._last_published.get(key) == raw:
            return
        msg = String()
        msg.data = raw
        publisher.publish(msg)
        self._last_published[key] = raw

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLAMissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
