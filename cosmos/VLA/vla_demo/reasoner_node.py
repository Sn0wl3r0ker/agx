from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .json_utils import dumps_json, loads_json
from .topics import (
    ANNOTATED_IMAGE_TOPIC,
    CAMERA_IMAGE_TOPIC,
    CURRENT_PROMPT_TOPIC,
    INFERENCE_INTERVAL_TOPIC,
    INFERENCE_RESULT_TOPIC,
)


def _extract_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            payload, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in reasoner output")


class MockReasonerBackend:
    def infer(self, prompt_payload: dict[str, Any], frames: list[np.ndarray]) -> dict[str, Any]:
        step_id = int(prompt_payload.get("step_id", -1))
        landmarks = list(prompt_payload.get("expected_landmarks", []))
        return {
            "step_id": step_id,
            "step_match": bool(frames),
            "step_completed": bool(frames),
            "observed_landmarks": landmarks,
            "confidence": 0.9 if frames else 0.0,
            "reason": "mock_backend",
        }


class CosmosCLIBackend:
    def __init__(
        self,
        python_executable: str,
        inference_script: str,
        host: str,
        port: int,
        model: str,
        timeout_sec: float,
        use_video: bool,
    ) -> None:
        self.python_executable = python_executable
        self.inference_script = inference_script
        self.host = host
        self.port = port
        self.model = model
        self.timeout_sec = timeout_sec
        self.use_video = use_video

    def infer(self, prompt_payload: dict[str, Any], frames: list[np.ndarray]) -> dict[str, Any]:
        if not frames:
            raise RuntimeError("Cosmos backend received no frames")
        with tempfile.TemporaryDirectory(prefix="vla_reasoner_") as tmpdir:
            media_path, media_flag = self._write_media(Path(tmpdir), frames)
            cmd = [
                self.python_executable,
                self.inference_script,
                "online",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--prompt",
                str(prompt_payload.get("prompt_text", "")),
            ]
            if self.model:
                cmd.extend(["--model", self.model])
            cmd.extend([media_flag, str(media_path)])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            payload = _extract_json_object(result.stdout)
            payload.setdefault("step_id", int(prompt_payload.get("step_id", -1)))
            return payload

    def _write_media(self, tmpdir: Path, frames: list[np.ndarray]) -> tuple[Path, str]:
        if self.use_video and len(frames) > 1:
            path = tmpdir / "clip.mp4"
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                8.0,
                (width, height),
            )
            for frame in frames:
                writer.write(frame)
            writer.release()
            return path, "--videos"
        path = tmpdir / "frame.jpg"
        cv2.imwrite(str(path), frames[-1])
        return path, "--images"


class CosmosReasonerNode(Node):
    def __init__(self) -> None:
        super().__init__("cosmos_reasoner_node")

        self.declare_parameter("backend_mode", "mock")
        self.declare_parameter("clip_seconds", 3.0)
        self.declare_parameter("python_executable", "python3")
        self.declare_parameter("inference_script", "/workspaces/cosmos_ws/inference.py")
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 8000)
        self.declare_parameter("model", "")
        self.declare_parameter("backend_timeout_sec", 90.0)
        self.declare_parameter("use_video", True)
        self.declare_parameter("annotate_image", True)

        self.bridge = CvBridge()
        self.frame_buffer: deque[tuple[float, np.ndarray]] = deque()
        self.prompt_payload: dict[str, Any] = {}
        self.inference_interval_sec = 1.5
        self.last_inference_time = 0.0
        self.last_annotation: dict[str, Any] | None = None
        self.last_frame: np.ndarray | None = None

        backend_mode = str(self.get_parameter("backend_mode").value)
        if backend_mode == "cosmos_cli":
            self.backend = CosmosCLIBackend(
                python_executable=str(self.get_parameter("python_executable").value),
                inference_script=str(self.get_parameter("inference_script").value),
                host=str(self.get_parameter("host").value),
                port=int(self.get_parameter("port").value),
                model=str(self.get_parameter("model").value),
                timeout_sec=float(self.get_parameter("backend_timeout_sec").value),
                use_video=bool(self.get_parameter("use_video").value),
            )
        else:
            self.backend = MockReasonerBackend()

        self.result_pub = self.create_publisher(String, INFERENCE_RESULT_TOPIC, 10)
        self.annotated_pub = self.create_publisher(Image, ANNOTATED_IMAGE_TOPIC, 10)
        self.create_subscription(Image, CAMERA_IMAGE_TOPIC, self._on_image, 10)
        self.create_subscription(String, CURRENT_PROMPT_TOPIC, self._on_prompt, 10)
        self.create_subscription(
            String, INFERENCE_INTERVAL_TOPIC, self._on_inference_interval, 10
        )
        self.create_timer(0.2, self._on_timer)

        self.get_logger().info(
            f"Cosmos reasoner ready in '{backend_mode}' mode. Subscribed to {CAMERA_IMAGE_TOPIC}."
        )

    def _on_prompt(self, msg: String) -> None:
        try:
            payload = loads_json(msg.data, "current_prompt")
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        if isinstance(payload, dict):
            self.prompt_payload = payload

    def _on_inference_interval(self, msg: String) -> None:
        try:
            payload = loads_json(msg.data, "inference_interval")
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        if isinstance(payload, dict):
            seconds = payload.get("seconds")
            if isinstance(seconds, (int, float)) and seconds > 0:
                self.inference_interval_sec = float(seconds)

    def _on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if stamp <= 0:
            stamp = self._now()
        self.frame_buffer.append((stamp, frame))
        self.last_frame = frame
        self._trim_buffer(stamp)
        if bool(self.get_parameter("annotate_image").value) and self.last_annotation is not None:
            self._publish_annotated_image(frame, self.last_annotation)

    def _on_timer(self) -> None:
        now = self._now()
        if not self.prompt_payload or not self.frame_buffer:
            return
        if now - self.last_inference_time < self.inference_interval_sec:
            return
        frames = [frame for _, frame in self.frame_buffer]
        try:
            payload = self.backend.infer(self.prompt_payload, frames)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Reasoner inference failed: {exc}")
            payload = {
                "step_id": int(self.prompt_payload.get("step_id", -1)),
                "step_match": False,
                "step_completed": False,
                "observed_landmarks": [],
                "confidence": 0.0,
                "reason": f"backend_error:{exc}",
            }
        if not isinstance(payload, dict):
            self.get_logger().warn("Reasoner backend returned a non-dict payload; coercing.")
            payload = {
                "step_id": int(self.prompt_payload.get("step_id", -1)),
                "step_match": False,
                "step_completed": False,
                "observed_landmarks": [],
                "confidence": 0.0,
                "reason": "invalid_backend_payload",
            }
        payload.setdefault("step_id", int(self.prompt_payload.get("step_id", -1)))
        payload.setdefault("observed_landmarks", [])
        payload.setdefault("reason", "reasoner")
        payload.setdefault("step_match", False)
        payload.setdefault("step_completed", False)
        payload.setdefault("confidence", 0.0)
        msg = String()
        msg.data = dumps_json(payload)
        self.result_pub.publish(msg)
        self.last_annotation = payload
        self.last_inference_time = now
        if self.last_frame is not None and bool(self.get_parameter("annotate_image").value):
            self._publish_annotated_image(self.last_frame, payload)

    def _publish_annotated_image(self, frame: np.ndarray, payload: dict[str, Any]) -> None:
        image = frame.copy()
        lines = [
            f"step_id={payload.get('step_id')}",
            f"match={payload.get('step_match')} complete={payload.get('step_completed')}",
            f"confidence={float(payload.get('confidence', 0.0)):.2f}",
            f"reason={payload.get('reason', '')}",
        ]
        y = 28
        for line in lines:
            cv2.putText(
                image,
                str(line),
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 255, 40),
                2,
                cv2.LINE_AA,
            )
            y += 28
        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.annotated_pub.publish(msg)

    def _trim_buffer(self, now: float) -> None:
        clip_seconds = float(self.get_parameter("clip_seconds").value)
        while self.frame_buffer and (now - self.frame_buffer[0][0]) > clip_seconds:
            self.frame_buffer.popleft()

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CosmosReasonerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
