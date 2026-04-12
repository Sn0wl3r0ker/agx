#!/usr/bin/env python3
"""
AGX ROS Dashboard - 零依賴多執行緒與日誌版
啟動: python3 dashboard.py [--port 8080]
"""

import http.server
import json
import subprocess
import os
import sys
import argparse
import time
import logging
import base64
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer  # 引入多執行緒伺服器

# --- 1. 日誌系統設定 (Logging) ---
logging.basicConfig(
    level=logging.DEBUG, # 開發期設為 DEBUG，上線可改為 INFO
    format='%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dashboard_debug.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("agx_dashboard")

# --- Configuration ---
PROJECT_NAME = "agx_ros"
SERVICES = {
    "planning":       {"name": "ROS 2 高階規劃",     "container": "planning",      "icon": "🧭", "service": "planning"},
    "foxglove":       {"name": "ROS 2 資料視覺化",   "container": "foxglove",      "icon": "📊", "service": "visualization"},
    "vlm":            {"name": "Isaac ROS 視覺加速", "container": "vlm",           "icon": "👁️", "service": "vlm"},
    "nanollm":        {"name": "Nano LLM",           "container": "nanollm",       "icon": "🤖", "service": "nanollm"},
    "cosmos":         {"name": "Cosmos-Reason2",     "container": "cosmos",        "icon": "🌌", "service": "cosmos"},
}

TASKS = {
    "planning": {
        "label": "Planning Tasks (ROS 2)",
        "container": "planning",
        "items": {
            "plan_lidar":    {"name": "Lidar 啟動測試", "cmd": "ros2 launch urg_node2 urg_node2.launch.py", "icon": "📡"},
            "plan_slam":     {"name": "SLAM Bringup", "cmd": "ros2 launch car_control slam_bringup.launch.py", "icon": "🗺️"},
            "plan_keyboard": {"name": "Keyboard Control", "cmd": "ros2 run teleop_twist_keyboard teleop_twist_keyboard", "icon": "⌨️"},
            "plan_savemap":  {"name": "Save Map", "cmd": "ros2 run nav2_map_server map_saver_cli -f /root/ros2_ws/src/car_control/config/my_map", "icon": "💾"},
        }
    }
}

# --- Detect environment ---
def detect_env():
    try:
        ctx = subprocess.check_output(["docker", "context", "show"], text=True).strip()
        if "agx" in ctx: return "agx", ".env.agx"
    except Exception: pass
    try:
        arch = subprocess.check_output(["uname", "-m"], text=True).strip()
        if arch == "aarch64": return "agx", ".env.agx"
    except Exception: pass
    return "pc", ".env"

MODE, ENV_FILE = detect_env()
COMPOSE_FILE = "docker-compose.pc.yaml" if MODE == "pc" else "docker-compose.yaml"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("系統環境偵測: Mode=%s, EnvFile=%s", MODE, ENV_FILE)

# --- ROS 2 Publisher Process ---
cmd_process_pipe = None

PY_PUB_SCRIPT = """
import rclpy
from geometry_msgs.msg import Twist
import sys
import json
rclpy.init()
node = rclpy.create_node('web_teleop')
pub = node.create_publisher(Twist, '/cmd_vel', 10)
t = Twist()
for line in sys.stdin:
    try:
        d = json.loads(line)
        t.linear.x = float(d.get('lx', 0.0))
        t.angular.z = float(d.get('az', 0.0))
        pub.publish(t)
    except Exception:
        pass
"""

def ensure_cmd_process():
    global cmd_process_pipe
    if cmd_process_pipe is None or cmd_process_pipe.poll() is not None:
        logger.info("準備啟動持久化 ROS 2 Publisher Process...")
        encoded = base64.b64encode(PY_PUB_SCRIPT.encode('utf-8')).decode('utf-8')
        cmd = [
            'docker', 'exec', '-i', 'planning', 'bash', '-c',
            f'source /opt/ros/humble/setup.bash && python3 -c "$(echo {encoded} | base64 -d)"'
        ]
        try:
            cmd_process_pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
            logger.info("持久化 ROS 2 Publisher Process 啟動成功")
        except Exception as e:
            logger.error("啟動 Publisher Process 失敗: %s", str(e))

def run_cmd(cmd, timeout=30):
    logger.debug("執行指令: %s", cmd)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=PROJECT_DIR
        )
        if result.returncode != 0:
            logger.warning("指令結束但有錯誤輸出 (Code %d): %s", result.returncode, result.stderr.strip())
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        logger.error("指令執行超時 (%d秒): %s", timeout, cmd)
        return {"ok": False, "stdout": "", "stderr": "Command timed out"}
    except Exception as e:
        logger.error("執行指令發生例外錯誤: %s", str(e))
        return {"ok": False, "stdout": "", "stderr": str(e)}

def get_container_status():
    result = run_cmd(
        f'docker ps -a --filter "label=com.docker.compose.project={PROJECT_NAME}" '
        f'--format "{{{{.Names}}}}|{{{{.Status}}}}|{{{{.State}}}}"'
    )
    statuses = {}
    if result["ok"] and result["stdout"]:
        for line in result["stdout"].splitlines():
            parts = line.split("|")
            if len(parts) >= 3:
                statuses[parts[0]] = {"status": parts[1], "state": parts[2]}
    return statuses

def get_tmux_sessions():
    result = run_cmd('tmux ls -F "#{session_name}" 2>/dev/null')
    if result["ok"] and result["stdout"]:
        return [s for s in result["stdout"].splitlines() if s.startswith("agx_") or s.startswith("plan_")]
    return []

def compose_cmd_for():
    return (f'docker compose --env-file {os.path.join(PROJECT_DIR, ENV_FILE)} '
            f'-f {os.path.join(PROJECT_DIR, COMPOSE_FILE)} -p {PROJECT_NAME}')

# --- API Handlers ---
def handle_api(path, params):
    action = path.replace("/api/", "")
    logger.debug("收到 API 請求: action=%s, params=%s", action, params)

    if action == "status":
        return {"containers": get_container_status(), "sessions": get_tmux_sessions(), "mode": MODE}

    elif action == "service/up":
        service_id = params.get("folder", [None])[0]
        if not service_id or service_id not in SERVICES: return {"ok": False, "error": "Invalid service"}
        return run_cmd(f'{compose_cmd_for()} up -d {SERVICES[service_id].get("service", service_id)}', timeout=120)

    elif action == "service/down":
        service_id = params.get("folder", [None])[0]
        if not service_id or service_id not in SERVICES: return {"ok": False, "error": "Invalid service"}
        return run_cmd(f'{compose_cmd_for()} rm -s -v -f {SERVICES[service_id].get("service", service_id)}', timeout=60)

    elif action == "service/build":
        service_id = params.get("folder", [None])[0]
        if not service_id or service_id not in SERVICES: return {"ok": False, "error": "Invalid service"}
        return run_cmd(f'{compose_cmd_for()} build {SERVICES[service_id].get("service", service_id)}', timeout=600)

    elif action == "service/rebuild":
        service_id = params.get("folder", [None])[0]
        if not service_id or service_id not in SERVICES: return {"ok": False, "error": "Invalid service"}
        return run_cmd(f'{compose_cmd_for()} up -d --build --force-recreate {SERVICES[service_id].get("service", service_id)}', timeout=600)

    elif action == "service/all-up":
        return run_cmd(f'docker compose --env-file {ENV_FILE} -p {PROJECT_NAME} up -d', timeout=120)

    elif action == "service/all-down":
        return run_cmd(f'docker compose --env-file {ENV_FILE} -p {PROJECT_NAME} down --remove-orphans', timeout=60)

    elif action == "task/launch":
        task_id = params.get("task", [None])[0]
        group = params.get("group", [None])[0]
        if not task_id or not group or group not in TASKS: return {"ok": False, "error": "Invalid task"}
        task_info = TASKS[group]["items"].get(task_id)
        if not task_info: return {"ok": False, "error": "Task not found"}
        
        if task_id in get_tmux_sessions(): return {"ok": True, "stdout": f"Task '{task_id}' already running."}
        
        container = TASKS[group]["container"]
        run_cmd(f'tmux new-session -d -s {task_id}')
        time.sleep(0.5)
        run_cmd(f'tmux send-keys -t {task_id}:0 "docker exec -it {container} bash -ic \'{task_info["cmd"]}\'" C-m')
        logger.info("啟動 Tmux 任務: %s", task_id)
        return {"ok": True, "stdout": f"Task '{task_id}' launched."}

    elif action == "task/stop":
        task_id = params.get("task", [None])[0]
        if task_id == "all":
            sessions = get_tmux_sessions()
            for s in sessions: run_cmd(f'tmux kill-session -t {s}')
            logger.info("停止所有 Tmux 任務")
            return {"ok": True, "stdout": f"Stopped {len(sessions)} tasks."}
        if not task_id: return {"ok": False, "error": "No task specified"}
        logger.info("停止 Tmux 任務: %s", task_id)
        return run_cmd(f'tmux kill-session -t {task_id}')

    elif action == "task/logs":
        task_id = params.get("task", [None])[0]
        if not task_id: return {"ok": False, "error": "Task not specified"}
        return run_cmd(f'tmux capture-pane -t {task_id} -p -S -50 2>/dev/null')

    elif action == "logs":
        folder = params.get("folder", [None])[0]
        lines = params.get("lines", ["50"])[0]
        if folder and folder in SERVICES:
            return run_cmd(f'docker logs --tail {lines} {SERVICES[folder]["container"]}', timeout=10)
        return run_cmd(f'docker compose --env-file {ENV_FILE} -p {PROJECT_NAME} logs --tail {lines}', timeout=10)

    elif action == "cmd_vel":
        lx = params.get("lx", ["0.0"])[0]
        az = params.get("az", ["0.0"])[0]
        ensure_cmd_process()
        global cmd_process_pipe
        if cmd_process_pipe and cmd_process_pipe.poll() is None:
            try:
                cmd_process_pipe.stdin.write(json.dumps({"lx": lx, "az": az}) + "\n")
                cmd_process_pipe.stdin.flush()
                return {"ok": True}
            except Exception as e:
                logger.error("Pipe 寫入錯誤: %s", str(e))
                return {"ok": False, "error": str(e)}
        return {"ok": False, "error": "Planning container not running"}

    return {"ok": False, "error": "Unknown action"}


# --- HTTP Server Handler ---
class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path.startswith("/api/"):
            result = handle_api(path, params)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode())
        else:
            html = DASHBOARD_HTML.replace(
                "__SERVICES_JSON__", json.dumps(SERVICES, ensure_ascii=False)
            ).replace(
                "__TASKS_JSON__", json.dumps(TASKS, ensure_ascii=False)
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

    def log_message(self, format, *args):
        # 覆寫預設日誌，交給我們的 logging 模組統一管理
        logger.debug("%s - %s", self.client_address[0], format % args)

# --- HTML 原始碼 (與原始版本幾乎相同，僅保留 fetch) ---
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AGX ROS Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
/* 樣式保持不變 */
:root {
  --bg-primary: #0a0e1a; --bg-secondary: #111827; --bg-card: #1a2035; --bg-card-hover: #1f2847;
  --border: #2a3555; --text-primary: #e8eaf0; --text-secondary: #8892a8; --text-muted: #5a6480;
  --accent-blue: #3b82f6; --accent-blue-glow: rgba(59, 130, 246, 0.3);
  --accent-green: #22c55e; --accent-green-glow: rgba(34, 197, 94, 0.2);
  --accent-red: #ef4444; --accent-red-glow: rgba(239, 68, 68, 0.2);
  --accent-yellow: #f59e0b; --accent-yellow-glow: rgba(245, 158, 11, 0.2);
  --accent-purple: #a855f7; --accent-purple-glow: rgba(168, 85, 247, 0.2);
  --radius: 12px; --radius-sm: 8px; --shadow: 0 4px 24px rgba(0,0,0,0.3); --transition: all 0.2s ease;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; background: var(--bg-primary); color: var(--text-primary); min-height: 100vh; overflow-x: hidden; }
.header { background: linear-gradient(135deg, var(--bg-secondary) 0%, #0f172a 100%); border-bottom: 1px solid var(--border); padding: 16px 32px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; backdrop-filter: blur(12px); }
.header h1 { font-size: 20px; font-weight: 700; background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: flex; align-items: center; gap: 10px; }
.header h1 span { font-size: 24px; }
.mode-badge { font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 20px; text-transform: uppercase; letter-spacing: 1px; }
.mode-badge.pc { background: var(--accent-blue-glow); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
.mode-badge.agx { background: var(--accent-purple-glow); color: var(--accent-purple); border: 1px solid var(--accent-purple); }
.header-right { display: flex; align-items: center; gap: 12px; }
.refresh-indicator { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--accent-green-glow); } 50% { opacity: 0.7; box-shadow: 0 0 0 6px transparent; } }
.container { max-width: 1400px; margin: 0 auto; padding: 24px 32px; }
.section-title { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
.section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.service-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; margin-bottom: 32px; }
.service-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; transition: var(--transition); position: relative; overflow: hidden; }
.service-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)); opacity: 0; transition: opacity 0.3s; }
.service-card:hover { background: var(--bg-card-hover); border-color: var(--accent-blue); }
.service-card:hover::before { opacity: 1; }
.service-card.running { border-color: var(--accent-green); }
.service-card.running::before { background: var(--accent-green); opacity: 1; }
.service-card.stopped { border-color: var(--border); }
.card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.card-title { display: flex; align-items: center; gap: 10px; }
.card-icon { font-size: 24px; }
.card-name { font-weight: 600; font-size: 15px; }
.card-folder { font-size: 12px; color: var(--text-muted); font-family: monospace; }
.status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.status-dot.running { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green-glow); }
.status-dot.exited { background: var(--accent-red); }
.card-status { font-size: 12px; color: var(--text-secondary); margin-bottom: 16px; min-height: 18px; }
.card-actions { display: flex; gap: 8px; flex-wrap: wrap; }
.btn { padding: 6px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border); background: var(--bg-secondary); color: var(--text-primary); font-size: 12px; font-weight: 500; font-family: 'Inter', sans-serif; cursor: pointer; transition: var(--transition); display: inline-flex; align-items: center; gap: 5px; }
.btn:hover { border-color: var(--accent-blue); background: rgba(59,130,246,0.1); }
.btn:active { transform: scale(0.96); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn.btn-green { border-color: var(--accent-green); color: var(--accent-green); }
.btn.btn-green:hover { background: var(--accent-green-glow); }
.btn.btn-red { border-color: var(--accent-red); color: var(--accent-red); }
.btn.btn-red:hover { background: var(--accent-red-glow); }
.btn.btn-yellow { border-color: var(--accent-yellow); color: var(--accent-yellow); }
.btn.btn-yellow:hover { background: var(--accent-yellow-glow); }
.btn.btn-primary { background: linear-gradient(135deg, var(--accent-blue), #2563eb); border: none; color: white; padding: 8px 20px; font-size: 13px; }
.btn.btn-primary:hover { filter: brightness(1.15); }
.bulk-bar { display: flex; gap: 10px; margin-bottom: 24px; flex-wrap: wrap; }
.task-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-bottom: 16px; }
.task-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 14px 18px; display: flex; align-items: center; justify-content: space-between; transition: var(--transition); }
.task-card:hover { border-color: var(--accent-blue); background: var(--bg-card-hover); }
.task-card.active { border-color: var(--accent-green); }
.task-info { display: flex; align-items: center; gap: 10px; }
.task-icon { font-size: 20px; }
.task-name { font-size: 13px; font-weight: 500; }
.task-cmd { font-size: 11px; color: var(--text-muted); font-family: monospace; margin-top: 2px; }
.task-actions { display: flex; gap: 6px; }
.log-viewer { background: #000; border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; margin-top: 16px; max-height: 300px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.6; color: #a0f0a0; white-space: pre-wrap; word-break: break-all; display: none; }
.log-viewer.visible { display: block; }
.toast-container { position: fixed; bottom: 24px; right: 24px; display: flex; flex-direction: column; gap: 8px; z-index: 1000; }
.toast { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 12px 20px; font-size: 13px; box-shadow: var(--shadow); animation: slideIn 0.3s ease; max-width: 400px; }
.toast.success { border-left: 3px solid var(--accent-green); }
.toast.error { border-left: 3px solid var(--accent-red); }
.toast.info { border-left: 3px solid var(--accent-blue); }
@keyframes slideIn { from { transform: translateX(100px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
.sessions-bar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 16px; min-height: 32px; }
.session-tag { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; background: var(--accent-green-glow); border: 1px solid var(--accent-green); font-size: 12px; color: var(--accent-green); font-weight: 500; }
.session-tag .kill-btn { cursor: pointer; opacity: 0.6; transition: opacity 0.2s; font-size: 14px; }
.session-tag .kill-btn:hover { opacity: 1; }
.no-sessions { font-size: 13px; color: var(--text-muted); }
.controller-panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 24px; text-align: center; }
.controller-info { font-size: 13px; color: var(--text-secondary); margin-bottom: 16px; }
.controller-grid { display: inline-grid; grid-template-columns: repeat(3, 64px); grid-template-rows: repeat(3, 64px); gap: 8px; margin-bottom: 16px; }
.ctrl-btn { width: 64px; height: 64px; border-radius: var(--radius-sm); border: 2px solid var(--border); background: var(--bg-secondary); color: var(--text-primary); font-size: 18px; font-weight: 700; font-family: 'Inter', sans-serif; cursor: pointer; transition: var(--transition); display: flex; flex-direction: column; align-items: center; justify-content: center; user-select: none; -webkit-user-select: none; }
.ctrl-btn:hover { border-color: var(--accent-blue); background: rgba(59,130,246,0.1); }
.ctrl-btn:active, .ctrl-btn.active { background: var(--accent-blue); border-color: var(--accent-blue); color: white; transform: scale(0.95); }
.ctrl-btn.ctrl-stop { border-color: var(--accent-red); color: var(--accent-red); }
.ctrl-btn.ctrl-stop:active, .ctrl-btn.ctrl-stop.active { background: var(--accent-red); color: white; }
.ctrl-sub { font-size: 10px; font-weight: 400; opacity: 0.6; }
.speed-control { display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 12px; font-size: 13px; color: var(--text-secondary); }
.speed-control input[type=range] { width: 150px; accent-color: var(--accent-blue); }
.ctrl-status { font-size: 12px; color: var(--text-muted); font-family: monospace; }
@media (max-width: 768px) { .container { padding: 16px; } .service-grid { grid-template-columns: 1fr; } .header { padding: 12px 16px; } }
</style>
</head>
<body>

<div class="header">
  <h1><span>🚀</span> AGX ROS Dashboard</h1>
  <div class="header-right">
    <div class="refresh-indicator"></div>
    <span id="modeLabel" class="mode-badge pc">PC</span>
  </div>
</div>

<div class="container">
  <div class="section-title">Docker 服務</div>
  <div class="bulk-bar">
    <button class="btn btn-primary" onclick="allUp()">▶ 全部啟動</button>
    <button class="btn btn-red" onclick="allDown()">⏹ 全部停止</button>
    <button class="btn" onclick="refreshStatus()">🔄 重新整理</button>
    <button class="btn" onclick="toggleLogs()">📋 Logs</button>
  </div>
  <div class="service-grid" id="serviceGrid"></div>
  <div class="log-viewer" id="logViewer"></div>

  <div class="section-title" style="margin-top:24px;">Planning 任務 (ROS 2)</div>
  <div class="task-grid" id="planningTasks"></div>

  <div class="section-title" style="margin-top:24px;">🎮 虛擬搖桿控制</div>
  <div class="controller-panel">
    <div class="controller-info">按住 WASD 或點擊按鈕控制車輛 · 空白鍵急停</div>
    <div class="controller-grid">
      <div></div>
      <button class="ctrl-btn" data-dir="w" id="btnW">W<br><span class="ctrl-sub">前進</span></button>
      <div></div>
      <button class="ctrl-btn" data-dir="a" id="btnA">A<br><span class="ctrl-sub">左轉</span></button>
      <button class="ctrl-btn ctrl-stop" data-dir="stop" id="btnStop">⏹<br><span class="ctrl-sub">停止</span></button>
      <button class="ctrl-btn" data-dir="d" id="btnD">D<br><span class="ctrl-sub">右轉</span></button>
      <div></div>
      <button class="ctrl-btn" data-dir="s" id="btnS">S<br><span class="ctrl-sub">後退</span></button>
      <div></div>
    </div>
    <div class="speed-control">
      <label>速度: <input type="range" id="speedSlider" min="0.1" max="1.0" step="0.1" value="0.3"></label>
      <span id="speedValue">0.3 m/s</span>
    </div>
    <div class="ctrl-status" id="ctrlStatus">待命</div>
  </div>

  <div class="section-title" style="margin-top:24px;">執行中的 Tmux Sessions</div>
  <div class="sessions-bar" id="sessionsBar">
    <span class="no-sessions">載入中...</span>
  </div>
</div>

<div class="toast-container" id="toastContainer"></div>

<script>
const SERVICES = __SERVICES_JSON__;
const TASKS = __TASKS_JSON__;

let containerStatuses = {};
let activeSessions = [];

async function api(endpoint, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const url = `/api/${endpoint}${qs ? '?' + qs : ''}`;
  try {
    const r = await fetch(url);
    return await r.json();
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function renderServices() {
  const grid = document.getElementById('serviceGrid');
  grid.innerHTML = '';
  for (const [folder, svc] of Object.entries(SERVICES)) {
    const info = containerStatuses[svc.container];
    const state = info ? info.state : 'stopped';
    const statusText = info ? info.status : 'Not running';
    const card = document.createElement('div');
    card.className = `service-card ${state}`;
    card.innerHTML = `
      <div class="card-header">
        <div class="card-title"><span class="card-icon">${svc.icon}</span><div><div class="card-name">${svc.name}</div><div class="card-folder">${folder}</div></div></div>
        <div class="status-dot ${state}"></div>
      </div>
      <div class="card-status">${statusText}</div>
      <div class="card-actions">
        <button class="btn btn-green" onclick="serviceAction('up','${folder}')" ${state === 'running' ? 'disabled' : ''}>▶ Start</button>
        <button class="btn btn-red" onclick="serviceAction('down','${folder}')" ${state !== 'running' ? 'disabled' : ''}>⏹ Stop</button>
        <button class="btn btn-yellow" onclick="serviceAction('build','${folder}')">🔨 Build</button>
        <button class="btn" onclick="serviceAction('rebuild','${folder}')">♻️ Rebuild</button>
        <button class="btn" onclick="showLogs('${folder}')">📋</button>
      </div>
    `;
    grid.appendChild(card);
  }
}

function renderTasks() {
  for (const [group, data] of Object.entries(TASKS)) {
    const gridId = group + 'Tasks';
    const grid = document.getElementById(gridId);
    grid.innerHTML = '';
    for (const [taskId, task] of Object.entries(data.items)) {
      const isActive = activeSessions.includes(taskId);
      const card = document.createElement('div');
      card.className = `task-card ${isActive ? 'active' : ''}`;
      card.innerHTML = `
        <div class="task-info"><span class="task-icon">${task.icon}</span><div><div class="task-name">${task.name}</div><div class="task-cmd">${task.cmd.length > 50 ? task.cmd.substring(0, 50) + '...' : task.cmd}</div></div></div>
        <div class="task-actions">
          ${isActive ? `<button class="btn btn-red" onclick="taskAction('stop','${taskId}')">⏹</button><button class="btn" style="margin-left:4px" onclick="showTaskLogs('${taskId}')">📋 Logs</button>` : `<button class="btn btn-green" onclick="taskAction('launch','${taskId}','${group}')">▶ Start</button>`}
        </div>
      `;
      grid.appendChild(card);
    }
  }
}

function renderSessions() {
  const bar = document.getElementById('sessionsBar');
  if (activeSessions.length === 0) { bar.innerHTML = '<span class="no-sessions">沒有執行中的任務</span>'; return; }
  bar.innerHTML = activeSessions.map(s => `<span class="session-tag">${s}<span class="kill-btn" onclick="taskAction('stop','${s}')" title="Terminate">✕</span></span>`).join('') + `<button class="btn btn-red" style="margin-left:8px" onclick="taskAction('stop','all')">全部停止</button>`;
}

async function serviceAction(action, folder) {
  toast(`${action} ${folder}...`, 'info');
  const r = await api(`service/${action}`, { folder });
  toast(r.ok ? `${folder} ${action} 完成 ✓` : `失敗: ${r.stderr || r.error}`, r.ok ? 'success' : 'error');
  await refreshStatus();
}

async function allUp() { toast('啟動所有服務...', 'info'); await api('service/all-up'); await refreshStatus(); }
async function allDown() { toast('停止所有服務...', 'info'); await api('service/all-down'); await refreshStatus(); }

async function taskAction(action, task, group=null) {
  toast(`執行 ${action} ${task}...`, 'info');
  const r = await api(`task/${action}`, { task, group });
  toast(r.ok ? `${task} 操作成功 ✓` : `失敗: ${r.stderr || r.error}`, r.ok ? 'success' : 'error');
  await refreshStatus();
}

async function showLogs(folder) {
  const viewer = document.getElementById('logViewer');
  viewer.classList.add('visible');
  viewer.textContent = 'Loading...';
  const r = await api('logs', { folder, lines: 80 });
  viewer.textContent = r.stdout || r.stderr || 'No logs available';
  viewer.scrollTop = viewer.scrollHeight;
}

async function showTaskLogs(task) {
  const viewer = document.getElementById('logViewer');
  viewer.classList.add('visible');
  viewer.textContent = 'Loading...';
  const r = await api('task/logs', { task });
  viewer.textContent = r.stdout || r.stderr || 'No logs available';
  viewer.scrollTop = viewer.scrollHeight;
}

function toggleLogs() { const v = document.getElementById('logViewer'); v.classList.contains('visible') ? v.classList.remove('visible') : showLogs(''); }

async function refreshStatus() {
  const data = await api('status');
  containerStatuses = data.containers || {};
  activeSessions = data.sessions || [];
  const mLabel = document.getElementById('modeLabel');
  mLabel.textContent = (data.mode || 'pc').toUpperCase();
  mLabel.className = `mode-badge ${data.mode || 'pc'}`;
  renderServices(); renderTasks(); renderSessions();
}

refreshStatus();
setInterval(refreshStatus, 5000);

let ctrlInterval = null;
let currentDir = null;
const speedSlider = document.getElementById('speedSlider');
const speedValue = document.getElementById('speedValue');
const ctrlStatus = document.getElementById('ctrlStatus');

speedSlider.addEventListener('input', () => speedValue.textContent = speedSlider.value + ' m/s');

function getVel(dir) {
  const spd = parseFloat(speedSlider.value);
  const turnSpd = spd * 3;
  switch(dir) {
    case 'w': return { lx: spd, az: 0 };
    case 's': return { lx: -spd, az: 0 };
    case 'a': return { lx: 0, az: turnSpd };
    case 'd': return { lx: 0, az: -turnSpd };
    default:  return { lx: 0, az: 0 };
  }
}

// 由於後端改為多執行緒，這裡的 fetch 頻率可以稍微調高，不會卡死背景作業
function sendVel(dir) {
  const v = getVel(dir);
  ctrlStatus.textContent = `cmd_vel: linear.x=${v.lx.toFixed(2)}, angular.z=${v.az.toFixed(2)}`;
  fetch(`/api/cmd_vel?lx=${v.lx}&az=${v.az}`).catch(() => {});
}

function startControl(dir) {
  if (currentDir === dir) return;
  stopControl();
  currentDir = dir;
  const btn = document.querySelector(`[data-dir="${dir}"]`);
  if (btn) btn.classList.add('active');
  sendVel(dir);
  ctrlInterval = setInterval(() => sendVel(dir), 150); // 設定為 150ms 較為平衡
}

function stopControl() {
  if (ctrlInterval) clearInterval(ctrlInterval);
  ctrlInterval = null;
  currentDir = null;
  document.querySelectorAll('.ctrl-btn').forEach(b => b.classList.remove('active'));
  sendVel('stop');
  ctrlStatus.textContent = '待命';
}

document.querySelectorAll('.ctrl-btn').forEach(btn => {
  const dir = btn.dataset.dir;
  if (dir === 'stop') {
    btn.addEventListener('mousedown', stopControl);
    btn.addEventListener('touchstart', (e) => { e.preventDefault(); stopControl(); });
  } else {
    btn.addEventListener('mousedown', () => startControl(dir));
    btn.addEventListener('touchstart', (e) => { e.preventDefault(); startControl(dir); });
  }
  btn.addEventListener('mouseup', stopControl);
  btn.addEventListener('mouseleave', () => { if (currentDir === dir) stopControl(); });
  btn.addEventListener('touchend', stopControl);
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const key = e.key.toLowerCase();
  if (['w','a','s','d'].includes(key)) { e.preventDefault(); startControl(key); }
  if (key === ' ') { e.preventDefault(); stopControl(); }
});
document.addEventListener('keyup', (e) => {
  const key = e.key.toLowerCase();
  if (['w','a','s','d'].includes(key) && currentDir === key) stopControl();
});
</script>
</body>
</html>"""

def main():
    parser = argparse.ArgumentParser(description="AGX ROS Dashboard (Zero-Dependency Threaded)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    # 關鍵改變：使用 ThreadingHTTPServer 取代原本的 HTTPServer
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    
    logger.info("啟動 Dashboard:")
    logger.info("Mode: %s", MODE.upper())
    logger.info("Env:  %s", ENV_FILE)
    logger.info("URL:  http://%s:%s", "localhost" if args.host == "0.0.0.0" else args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard 收到終止訊號，準備關閉...")
        server.server_close()

if __name__ == "__main__":
    main()