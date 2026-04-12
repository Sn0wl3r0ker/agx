# VLA Demo ROS2 Package

`cosmos/VLA` 是這次 AMR Demo 的高階任務層 ROS2 package，套件名稱為 `vla_demo`。

這個資料夾的目標是把以下能力集中在同一套程式碼中：

- 任務載入與步驟管理
- 視覺步驟驗證 prompt 發布
- Cosmos-Reason2 推論結果回收
- 語意控制 primitive 轉換為 `/cmd_vel`
- Demo 過程中的 telemetry topic 發布

目前設計是「同一份 package，在不同 container 跑不同 node」：

- `nanollm`：負責 ROS2 orchestration 與前視相機
- `cosmos`：負責 Cosmos-Reason2 推論
- `planning`：負責底層移動控制與 `/cmd_vel`

## 1. 目錄說明

```text
cosmos/VLA/
├── config/
│   └── sample_mission.yaml      # 範例任務檔
├── launch/
│   ├── mission_stack.launch.py  # mission manager + camera
│   ├── reasoner.launch.py       # reasoner node
│   └── executor.launch.py       # executor node
├── test/
│   ├── test_mission_loader.py   # schema / mission 載入測試
│   └── test_state_machine.py    # 狀態機測試
├── vla_demo/
│   ├── camera_front_node.py     # 前視相機 node
│   ├── executor_node.py         # 語意 primitive -> /cmd_vel
│   ├── mission_manager.py       # 任務管理與 state machine
│   ├── mission_loader.py        # mission 檔案載入
│   ├── prompting.py             # reasoner prompt 組裝
│   ├── reasoner_node.py         # Cosmos 推論 node
│   ├── schemas.py               # mission / step schema
│   ├── state_machine.py         # 切步邏輯
│   └── topics.py                # topic 常數
├── package.xml
├── setup.cfg
└── setup.py
```

## 2. 節點說明

### `vla_mission_manager`

用途：

- 載入 mission YAML
- 維護 step-by-step 狀態機
- 發布目前步驟、prompt、控制命令、mission state
- 接收推論結果與底盤狀態，決定是否切步

主要邏輯：

- 連續 `votes_needed` 次 `step_completed=true` 才切下一步
- 若 `confidence` 低於門檻，則不切步
- 若步驟 timeout，則進入 `PAUSED` 或 fallback 狀態

### `cosmos_reasoner_node`

用途：

- 訂閱前視相機影像
- 維護最近幾秒的 frame buffer
- 接收目前步驟 prompt
- 呼叫推論 backend
- 發布 `/vla/inference_result` 與 `/vla/annotated_image`

支援兩種 backend：

- `mock`
  - 用來快速驗證 ROS2 topic 鏈是否正常
  - 不依賴真實 Cosmos server
- `cosmos_cli`
  - 會呼叫 `cosmos/inference.py` 的 `online` 路徑
  - 前提是 `cosmos` container 內已經有可用的 OpenAI-compatible inference server

### `vla_executor`

用途：

- 訂閱 `/vla/control_command`
- 將高階 primitive 轉為 `/cmd_vel`
- 定期發布 `/vla/robot_status`

目前支援的 primitive：

- `move_forward_until_recheck`
- `approach_target_zone`
- `turn_right_90`
- `turn_left_90`
- `stop_and_hold`

### `camera_front_node`

用途：

- 獨占前視 camera
- 發布 `/camera/front/image_raw`
- 避免多個 container 同時搶同一支 `/dev/video*`

## 3. Topic 一覽

### 控制與任務 topic

- `/vla/set_mission`
- `/vla/abort`
- `/vla/current_step`
- `/vla/current_prompt`
- `/vla/inference_interval`
- `/vla/control_command`
- `/vla/mission_state`

### 感知與推論 topic

- `/camera/front/image_raw`
- `/vla/inference_result`
- `/vla/annotated_image`

### 底盤回報 topic

- `/vla/robot_status`
- `/cmd_vel`

## 4. Mission 檔案格式

範例檔案：`config/sample_mission.yaml`

重要欄位如下：

- `mission_id`
- `mission_text`
- `environment_id`
- `camera_source`
- `inference_interval_sec`
- `steps[]`

每個 step 需要的欄位：

- `step_id`
- `instruction`
- `visual_goal`
- `expected_landmarks`
- `control_primitive`
- `votes_needed`
- `confidence_threshold`
- `min_dwell_sec`
- `timeout_sec`
- `fallback`

## 5. 使用前準備

### Docker / ROS2 前提

請先確保以下 container 已經建好並啟動：

- `planning`
- `nanollm`
- `cosmos`

本專案目前的掛載方式如下：

- `planning` 會把 `cosmos/VLA` 掛到 `/root/ros2_ws/src/vla_demo`
- `nanollm` 會把 `cosmos/VLA` 掛到 `/opt/vla_demo`
- `cosmos` 會把 `cosmos/VLA` 掛到 `/opt/vla_demo`

### 建議啟動方式

在 repo 根目錄：

```bash
# 啟動 planning / nanollm / cosmos 三個服務
make up
```

如果只想啟動特定服務，也可以用既有的 compose 流程。

## 6. 快速開始

下面是最推薦的啟動順序。

### Step 1: 在 `planning` 啟動 executor

進入 container：

```bash
# 進入 planning container
docker exec -it planning bash
```

啟動 executor：

```bash
# source ROS2 與 workspace
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash

# 啟動語意控制執行器，將 primitive 轉成 /cmd_vel
ros2 run vla_demo vla_executor
```

### Step 2: 在 `nanollm` 啟動 camera node

進入 container：

```bash
# 進入 nanollm container
docker exec -it nanollm bash
```

啟動 camera：

```bash
# source ROS2
if [ -f /opt/ros/install/setup.bash ]; then
  source /opt/ros/install/setup.bash
else
  source /opt/ros/humble/setup.bash
fi

# 將共享 package 加進 PYTHONPATH
export PYTHONPATH=/opt/vla_demo:$PYTHONPATH

# 啟動前視相機 node，預設使用 /dev/video0
python3 -m vla_demo.camera_front_node
```

如果要改 camera device：

```bash
# 範例：改用 /dev/video2
python3 -m vla_demo.camera_front_node --ros-args -p video_device:=/dev/video2
```

### Step 3: 在 `nanollm` 啟動 mission manager

開另一個 shell：

```bash
# 進入 nanollm container
docker exec -it nanollm bash
```

啟動 mission manager：

```bash
# source ROS2
if [ -f /opt/ros/install/setup.bash ]; then
  source /opt/ros/install/setup.bash
else
  source /opt/ros/humble/setup.bash
fi

# 將共享 package 加進 PYTHONPATH
export PYTHONPATH=/opt/vla_demo:$PYTHONPATH

# 啟動任務管理 node
python3 -m vla_demo.mission_manager
```

若要一啟動就自動載入 sample mission：

```bash
# autostart=true 時，會直接載入 default_mission_file
python3 -m vla_demo.mission_manager --ros-args \
  -p autostart:=true \
  -p default_mission_file:=/opt/vla_demo/config/sample_mission.yaml
```

### Step 4: 在 `cosmos` 啟動 reasoner

進入 container：

```bash
# 進入 cosmos container
docker exec -it cosmos bash
```

先跑 mock 版，確認 topic 鏈正常：

```bash
# source ROS2 與 Cosmos python 環境
source /opt/ros/humble/setup.bash
source /workspaces/cosmos_ws/.venv/bin/activate

# 將共享 package 加進 PYTHONPATH
export PYTHONPATH=/opt/vla_demo:$PYTHONPATH

# 啟動 mock reasoner
python3 -m vla_demo.reasoner_node --ros-args -p backend_mode:=mock
```

若要切到 `cosmos_cli`：

```bash
# 前提：你已經在 cosmos container 內啟動好可用的 inference server
source /opt/ros/humble/setup.bash
source /workspaces/cosmos_ws/.venv/bin/activate
export PYTHONPATH=/opt/vla_demo:$PYTHONPATH

# 啟動真實推論模式
python3 -m vla_demo.reasoner_node --ros-args \
  -p backend_mode:=cosmos_cli \
  -p host:=localhost \
  -p port:=8000
```

## 7. 送出任務

在任一個有 ROS2 環境的 shell 中執行：

```bash
# 載入 sample mission 檔案
ros2 topic pub --once /vla/set_mission std_msgs/msg/String \
  "{data: '/opt/vla_demo/config/sample_mission.yaml'}"
```

也可以直接送 JSON：

```bash
# 直接以 JSON 字串送出 mission
ros2 topic pub --once /vla/set_mission std_msgs/msg/String \
  "{data: '{\"mission_id\":\"demo_inline\",\"mission_text\":\"demo\",\"environment_id\":\"hallway\",\"camera_source\":\"/camera/front/image_raw\",\"steps\":[{\"step_id\":1,\"instruction\":\"向前\",\"visual_goal\":\"看到走廊\",\"control_primitive\":\"move_forward_until_recheck\",\"fallback\":\"pause\"}]}' }"
```

## 8. 常用監看指令

### 看目前任務狀態

```bash
# 看 mission manager 的整體狀態
ros2 topic echo /vla/mission_state
```

### 看目前步驟與 prompt

```bash
# 看目前 step payload
ros2 topic echo /vla/current_step

# 看送給 Cosmos 的 prompt payload
ros2 topic echo /vla/current_prompt
```

### 看推論結果

```bash
# 看 reasoner 輸出的 JSON 結果
ros2 topic echo /vla/inference_result
```

### 看底盤執行狀態

```bash
# 看 executor 回報的狀態
ros2 topic echo /vla/robot_status

# 看實際送到車子的速度命令
ros2 topic echo /cmd_vel
```

### 看 ROS graph 是否有接起來

```bash
# 列出所有相關 topic
ros2 topic list | grep vla

# 確認前視相機有在發布
ros2 topic hz /camera/front/image_raw
```

## 9. 中止任務

```bash
# 發送 abort 指令，mission manager 會進入 ABORTED
ros2 topic pub --once /vla/abort std_msgs/msg/String "{data: 'operator_abort'}"
```

## 10. rosbag 錄製建議

若要做 teach run 或 replay 驗證，建議錄以下 topic：

```bash
# 建議錄製的 topic
ros2 bag record \
  /camera/front/image_raw \
  /vla/current_step \
  /vla/inference_result \
  /vla/mission_state \
  /vla/robot_status \
  /odom
```

## 11. 目前已實作的狀態

目前這個版本已完成：

- mission YAML 載入
- step schema 驗證
- state machine 切步邏輯
- mock reasoner topic 鏈
- executor primitive -> `/cmd_vel`
- annotated image topic

目前仍需要你現場補齊或確認：

- 真實 Cosmos inference server 的啟動方式
- 真機 camera device 編號
- `turn_right_90` 的角速度與持續時間校正
- 不同環境下的 confidence threshold 調整

## 12. 測試

本地可先跑：

```bash
# 語法檢查
python3 -m compileall -q /home/syslabros/agx/cosmos/VLA

# 單元測試
PYTHONPATH=/home/syslabros/agx/cosmos/VLA \
python3 -m unittest discover -s /home/syslabros/agx/cosmos/VLA/test
```

## 13. 常見問題

### `planning` 找不到 `vla_demo`

請確認：

- `planning/docker-compose.yaml` 已經把 `../cosmos/VLA` 掛到 `/root/ros2_ws/src/vla_demo`
- container 啟動後有重新 build workspace

可在 `planning` container 內手動重編：

```bash
source /opt/ros/humble/setup.bash
cd /root/ros2_ws
colcon build --symlink-install
source /root/ros2_ws/install/setup.bash
```

### `nanollm` 或 `cosmos` 找不到 `vla_demo`

請確認有執行：

```bash
# 將共享套件路徑加入 Python 匯入路徑
export PYTHONPATH=/opt/vla_demo:$PYTHONPATH
```

### reasoner 沒有輸出

請依序檢查：

- `/camera/front/image_raw` 是否有畫面
- `/vla/current_prompt` 是否有資料
- `backend_mode` 是否正確
- 若使用 `cosmos_cli`，server 是否真的已啟動

---

如果你接下來要把這份 README 再往 demo 操作手冊方向補，我建議下一步可以加：

- Foxglove 畫面配置範例
- 真機彩排 SOP
- `cosmos_cli` 實際 server 啟動命令
