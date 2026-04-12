## webui
`cd webui_system && uvicorn backend.app:app --host 0.0.0.0 --port 8089`

## vllm ros2 bridge
`python bridge_ws/vllm_ros2_bridge.py`

```bash
ros2 topic pub /llm/request std_msgs/msg/String "{data: '請描述這段影片中的人物、動作與是否有異常事件。'}" --once
ros2 topic pub /llm/video_uri std_msgs/msg/String "{data: 'file:///workspaces/cosmos_ws/video/sec_fly.mp4'}" --once
ros2 topic pub /llm/request std_msgs/msg/String "{data: '請描述路上特徵，與是否有人在草皮上跌倒(異常事件)。'}" --once
ros2 topic echo /llm/response
```

`ros2 topic pub /llm/event_flag std_msgs/msg/Bool "{data: true}" -1`