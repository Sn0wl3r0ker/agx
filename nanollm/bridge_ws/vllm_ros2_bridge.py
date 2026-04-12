#!/usr/bin/env python3

import json
import threading
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VLLMBridgeNode(Node):
    def __init__(self):
        super().__init__("vllm_ros2_bridge")

        self.declare_parameter("api_url", "http://127.0.0.1:8000/v1/chat/completions")
        self.declare_parameter("model_name", "/data/models/Cosmos-Reason2-2B")
        self.declare_parameter("default_system_prompt", "You are a helpful assistant.")

        self.api_url = self.get_parameter("api_url").get_parameter_value().string_value
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.default_system_prompt = (
            self.get_parameter("default_system_prompt").get_parameter_value().string_value
        )

        self.system_prompt = self.default_system_prompt
        self.video_uri = "file:///workspaces/cosmos_ws/video/sec_fly.mp4"
        self.lock = threading.Lock()

        self.system_prompt_sub = self.create_subscription(
            String, "/llm/system_prompt", self.system_prompt_callback, 10
        )
        self.video_uri_sub = self.create_subscription(
            String, "/llm/video_uri", self.video_uri_callback, 10
        )
        self.request_sub = self.create_subscription(
            String, "/llm/request", self.request_callback, 10
        )

        self.response_pub = self.create_publisher(String, "/llm/response", 10)
        self.status_pub = self.create_publisher(String, "/llm/status", 10)

        self.get_logger().info("vLLM ROS2 bridge started")
        self.get_logger().info(f"API URL: {self.api_url}")
        self.get_logger().info(f"Model: {self.model_name}")

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def system_prompt_callback(self, msg: String):
        new_prompt = msg.data.strip()
        if not new_prompt:
            return
        with self.lock:
            self.system_prompt = new_prompt
        self.get_logger().info(f"Updated system prompt: {new_prompt}")
        self.publish_status("system_prompt_updated")

    def video_uri_callback(self, msg: String):
        new_video_uri = msg.data.strip()
        if not new_video_uri:
            return
        with self.lock:
            self.video_uri = new_video_uri
        self.get_logger().info(f"Updated video uri: {new_video_uri}")
        self.publish_status("video_uri_updated")

    def request_callback(self, msg: String):
        user_text = msg.data.strip()
        if not user_text:
            return

        with self.lock:
            current_system_prompt = self.system_prompt
            current_video_uri = self.video_uri

        if not current_video_uri:
            self.publish_status("error:no_video_uri")
            self.get_logger().error("No video URI set. Publish /llm/video_uri first.")
            return

        try:
            result_text = self.call_vllm(
                system_prompt=current_system_prompt,
                user_text=user_text,
                video_uri=current_video_uri,
            )

            out_msg = String()
            out_msg.data = result_text
            self.response_pub.publish(out_msg)
            self.publish_status("inference_done")

        except Exception as e:
            err_text = f"[vLLM API error] {str(e)}"
            self.get_logger().error(err_text)
            out_msg = String()
            out_msg.data = err_text
            self.response_pub.publish(out_msg)
            self.publish_status(err_text)

    def call_vllm(self, system_prompt: str, user_text: str, video_uri: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是戶外機器人影片分析助理。"
                        "一律使用繁體中文。"
                        "不得輸出英文，除非使用者明確要求。"
                        "請依照指定格式完整回答。"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "請分析這段影片，並嚴格依照以下格式輸出：\n"
                                "1. 場景摘要\n"
                                "2. 主要物件與位置\n"
                                "3. 動作流程：列出3點，依時間順序\n"
                                "4. 異常事件：有人跌倒\n"
                                "5. 導航機器人的後續動作：列出3點\n"
                            )
                        },
                        {"type": "video_url", "video_url": {"url": video_uri}},
                    ],
                },
            ],
            "temperature": 0.3,
            "top_p": 0.85,
            "max_completion_tokens": 4096,
            "min_tokens": 256,
            "stream": False,
            # "mm_processor_kwargs": {
            #     "fps": 4,
            #     "do_sample_frames": True
            # }
        }

        response = requests.post(
            self.api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


def main(args=None):
    rclpy.init(args=args)
    node = VLLMBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()