#!/bin/bash
# ==============================================================================
#  客製化 Python Zenoh Bridge 連通測試 (PC 模式)
# ==============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

TIMEOUT=15
ROSCORE=bridge_test_roscore
BRIDGE=bridge_test_bridge
PLANNING=bridge_test_planning

echo "=========================================="
echo " ROS1 <-> ROS2 Custom Python Zenoh Bridge"
echo "=========================================="

echo ""
echo -e "${YELLOW}[0] 檢查容器狀態...${NC}"
for c in $ROSCORE $BRIDGE $PLANNING; do
    if docker ps --format '{{.Names}}' | grep -q "^${c}$"; then
        echo -e "  ${GREEN}✓${NC} ${c} is running"
    else
        echo -e "  ${RED}✗${NC} ${c} is NOT running"
        exit 1
    fi
done

# --- 1. roscore 連通 ---
echo ""
echo -e "${YELLOW}[1] 測試 roscore...${NC}"
if docker exec $ROSCORE bash -c "source /opt/ros/noetic/setup.bash && rostopic list" >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} roscore OK"
else
    echo -e "  ${RED}✗${NC} roscore 無回應"
    exit 1
fi

# --- 2. ROS2 topic list ---
echo ""
echo -e "${YELLOW}[2] 測試 ROS2 側 (planning container) 環境...${NC}"
ROS2_SETUP="source /opt/ros/humble/setup.bash 2>/dev/null"
if docker exec $PLANNING bash -c "${ROS2_SETUP} && ros2 topic list" >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} ROS2 環境可用"
else
    echo -e "  ${RED}✗${NC} ROS2 環境不可用"
    exit 1
fi

echo ""
echo -e "${CYAN}等待 Custom Bridge 載入套件與啟動 (10s)...${NC}"
sleep 10

# --- 3. ROS1 -> ROS2 ---
echo ""
echo -e "${YELLOW}[3] 測試 ROS1 -> ROS2 ...${NC}"
TEST_TOPIC="/bridge_test_r1_to_r2"
TEST_MSG="hello_zenoh_r1_$(date +%s)"

docker exec $PLANNING bash -c "rm -f /tmp/test_r2.log" 2>/dev/null || true

# 在 ROS2 端背景接收
docker exec -d $PLANNING bash -c "
    ${ROS2_SETUP}
    timeout ${TIMEOUT} ros2 topic echo --once ${TEST_TOPIC} std_msgs/msg/String > /tmp/test_r2.log 2>&1
"
sleep 5

# 在 ROS1 端發送
docker exec $ROSCORE bash -c "
    source /opt/ros/noetic/setup.bash
    timeout 3 rostopic pub -r 5 ${TEST_TOPIC} std_msgs/String \"data: '${TEST_MSG}'\"
" > /dev/null 2>&1 &
PUB_PID=$!

sleep $((TIMEOUT))
wait $PUB_PID 2>/dev/null || true

R2_RESULT=$(docker exec $PLANNING cat /tmp/test_r2.log 2>/dev/null || true)
if echo "$R2_RESULT" | grep -q "${TEST_MSG}"; then
    echo -e "  ${GREEN}✓ ROS1 -> ROS2 橋接成功！${NC}"
else
    echo -e "  ${RED}✗ ROS1 -> ROS2 失敗${NC}"
    echo "  ROS2 側 log:"
    echo "$R2_RESULT"
    echo "  Bridge logs:"
    docker logs --tail 5 $BRIDGE
    docker logs --tail 5 $PLANNING
fi

# --- 4. ROS2 -> ROS1 ---
echo ""
echo -e "${YELLOW}[4] 測試 ROS2 -> ROS1 ...${NC}"
TEST_TOPIC2="/bridge_test_r2_to_r1"
TEST_MSG2="hello_zenoh_r2_$(date +%s)"

docker exec $ROSCORE bash -c "rm -f /tmp/test_r1.log" 2>/dev/null || true

# 在 ROS1 端背景接收
docker exec -d $ROSCORE bash -c "
    source /opt/ros/noetic/setup.bash
    timeout ${TIMEOUT} rostopic echo -n 1 ${TEST_TOPIC2} > /tmp/test_r1.log 2>&1
"
sleep 5

# 在 ROS2 端發送
docker exec $PLANNING bash -c "
    ${ROS2_SETUP}
    timeout 5 ros2 topic pub --once ${TEST_TOPIC2} std_msgs/msg/String \"data: '${TEST_MSG2}'\"
" > /dev/null 2>&1 &
PUB_PID2=$!

sleep $((TIMEOUT))
wait $PUB_PID2 2>/dev/null || true

R1_RESULT=$(docker exec $ROSCORE cat /tmp/test_r1.log 2>/dev/null || true)
if echo "$R1_RESULT" | grep -q "${TEST_MSG2}"; then
    echo -e "  ${GREEN}✓ ROS2 -> ROS1 橋接成功！${NC}"
else
    echo -e "  ${RED}✗ ROS2 -> ROS1 失敗${NC}"
    echo "  ROS1 側 log:"
    echo "$R1_RESULT"
    echo "  Bridge logs:"
    docker logs --tail 5 $BRIDGE
    docker logs --tail 5 $PLANNING
fi

echo ""
echo "=========================================="
echo " 測試結束"
echo "=========================================="
echo ""
