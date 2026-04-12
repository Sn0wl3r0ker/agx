import asyncio
import json
import re
import threading
from pathlib import Path
from typing import Optional

import rclpy
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rclpy.node import Node
from std_msgs.msg import Bool, String


# =========================
# 工具函式
# =========================
def extract_location(text: str) -> str:
    known_places = [
        "後花園", "前庭", "大門口", "停車場", "中庭",
        "行政大樓", "實驗室", "倉庫", "操場", "屋頂"
    ]
    for place in known_places:
        if place in text:
            return place

    match = re.search(r"到(.+?)(去|做|進行|巡檢|查看|確認|$)", text)
    if match:
        return match.group(1).strip()

    return "未指定區域"


def pick_asset(text: str) -> str:
    if any(k in text for k in ["巡檢", "空拍", "俯視", "查看上方"]):
        return "UAV"
    if any(k in text for k in ["前往", "走過去", "到現場", "派遣地面車"]):
        return "UGV"
    return "UAV"


def build_fixed_reply(user_text: str) -> dict:
    location = extract_location(user_text)
    asset = pick_asset(user_text)
    reply = f"任務啟動，出動設備：{asset}，地點：{location}"
    return {
        "raw_text": user_text,
        "asset": asset,
        "target_location": location,
        "reply": reply,
    }


def resolve_frontend_dir() -> Path:
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir / "frontend",
        current_dir.parent / "frontend",
        current_dir,
    ]
    for path in candidates:
        if (path / "index.html").exists():
            return path
    return current_dir


# =========================
# WebSocket 管理
# =========================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, payload: dict):
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# =========================
# ROS2 Bridge Node
# =========================
class WebUIROSBridge(Node):
    def __init__(self, loop: asyncio.AbstractEventLoop, manager: ConnectionManager):
        super().__init__("webui_backend_bridge")

        self.loop = loop
        self.manager = manager

        self.current_system_prompt = ""
        self.current_video_uri = ""

        # Publishers
        self.request_pub = self.create_publisher(String, "/llm/request", 10)
        self.response_pub = self.create_publisher(String, "/llm/response", 10)
        self.status_pub = self.create_publisher(String, "/llm/status", 10)
        self.system_prompt_pub = self.create_publisher(String, "/llm/system_prompt", 10)
        self.video_uri_pub = self.create_publisher(String, "/llm/video_uri", 10)

        # Subscribers
        self.create_subscription(String, "/llm/response", self.on_response, 10)
        self.create_subscription(String, "/llm/status", self.on_status, 10)
        self.create_subscription(String, "/llm/system_prompt", self.on_system_prompt, 10)
        self.create_subscription(String, "/llm/video_uri", self.on_video_uri, 10)

        # 異常事件 flag
        self.create_subscription(Bool, "/llm/event_flag", self.on_event_flag, 10)

        self.get_logger().info("WebUI ROS bridge started.")
        self.get_logger().info("Subscribed: /llm/response /llm/status /llm/system_prompt /llm/video_uri /llm/event_flag")
        self.get_logger().info("Published : /llm/request /llm/response /llm/status /llm/system_prompt /llm/video_uri")

    def push_to_frontend(self, payload: dict):
        asyncio.run_coroutine_threadsafe(
            self.manager.broadcast_json(payload),
            self.loop
        )

    # ---------------------
    # ROS2 callbacks
    # ---------------------
    def on_response(self, msg: String):
        self.push_to_frontend({
            "type": "ai_reply",
            "text": msg.data
        })

    def on_status(self, msg: String):
        self.push_to_frontend({
            "type": "status",
            "text": msg.data
        })

    def on_system_prompt(self, msg: String):
        self.current_system_prompt = msg.data
        self.push_to_frontend({
            "type": "system_prompt",
            "text": msg.data
        })

    def on_video_uri(self, msg: String):
        self.current_video_uri = msg.data
        self.push_to_frontend({
            "type": "video_uri",
            "text": msg.data
        })

    def on_event_flag(self, msg: Bool):
        if not msg.data:
            return

        event_text = "異常事件觸發！事件：行人跌倒！派遣UGV中"
        status_text = "異常事件：行人跌倒，已派遣 UGV"

        response_msg = String()
        response_msg.data = event_text
        self.response_pub.publish(response_msg)

        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)

        self.push_to_frontend({
            "type": "event",
            "text": event_text,
            "event_type": "行人跌倒",
            "action": "派遣UGV中"
        })

    # ---------------------
    # 主流程
    # ---------------------
    def handle_user_command(self, user_text: str) -> dict:
        result = build_fixed_reply(user_text)

        request_msg = String()
        request_msg.data = user_text
        self.request_pub.publish(request_msg)

        response_msg = String()
        response_msg.data = result["reply"]
        self.response_pub.publish(response_msg)

        status_msg = String()
        status_msg.data = f"已建立任務：{result['asset']} -> {result['target_location']}"
        self.status_pub.publish(status_msg)

        return result

    def publish_system_prompt(self, text: str):
        self.current_system_prompt = text
        msg = String()
        msg.data = text
        self.system_prompt_pub.publish(msg)

    def publish_video_uri(self, text: str):
        self.current_video_uri = text
        msg = String()
        msg.data = text
        self.video_uri_pub.publish(msg)


# =========================
# FastAPI
# =========================
app = FastAPI()
manager = ConnectionManager()
FRONTEND_DIR = resolve_frontend_dir()

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

ros_node: Optional[WebUIROSBridge] = None
ros_thread: Optional[threading.Thread] = None


class CommandRequest(BaseModel):
    text: str


class TextRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event():
    global ros_node, ros_thread

    loop = asyncio.get_running_loop()

    rclpy.init(args=None)
    ros_node = WebUIROSBridge(loop=loop, manager=manager)

    def spin_ros():
        rclpy.spin(ros_node)

    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()


@app.on_event("shutdown")
async def shutdown_event():
    global ros_node
    if ros_node is not None:
        ros_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


@app.get("/")
async def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"index.html not found: {index_path}"}
        )
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    return {
        "ok": True,
        "frontend_dir": str(FRONTEND_DIR),
        "index_exists": (FRONTEND_DIR / "index.html").exists()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    await websocket.send_json({
        "type": "system",
        "text": "WebSocket 已連線，等待任務與事件。"
    })

    if ros_node is not None:
        await websocket.send_json({
            "type": "system_prompt",
            "text": ros_node.current_system_prompt
        })
        await websocket.send_json({
            "type": "video_uri",
            "text": ros_node.current_video_uri
        })

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/command")
async def api_command(req: CommandRequest):
    global ros_node
    if ros_node is None:
        return {"ok": False, "error": "ROS node not ready"}

    text = req.text.strip()
    if not text:
        return {"ok": False, "error": "empty text"}

    result = ros_node.handle_user_command(text)

    await manager.broadcast_json({
        "type": "task",
        "task": {
            "asset": result["asset"],
            "target_location": result["target_location"]
        }
    })

    return {
        "ok": True,
        "reply": result["reply"],
        "task": {
            "asset": result["asset"],
            "target_location": result["target_location"]
        }
    }


@app.post("/api/system_prompt")
async def api_system_prompt(req: TextRequest):
    global ros_node
    if ros_node is None:
        return {"ok": False, "error": "ROS node not ready"}

    ros_node.publish_system_prompt(req.text.strip())

    return {
        "ok": True,
        "text": req.text.strip()
    }


@app.post("/api/video_uri")
async def api_video_uri(req: TextRequest):
    global ros_node
    if ros_node is None:
        return {"ok": False, "error": "ROS node not ready"}

    ros_node.publish_video_uri(req.text.strip())

    return {
        "ok": True,
        "text": req.text.strip()
    }


@app.post("/api/simulate/event")
async def api_simulate_event():
    global ros_node
    if ros_node is None:
        return {"ok": False, "error": "ROS node not ready"}

    ros_node.on_event_flag(Bool(data=True))
    return {"ok": True}