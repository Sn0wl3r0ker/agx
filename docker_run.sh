#!/bin/bash

# ==========================================
# 1. 模式選擇與設定檔載入
# 使用方式: ./docker_run.sh [agx|pc]
# ==========================================

# 讀取第一個參數，預設為 "pc"
MODE=${1:-pc}

if [ "$MODE" == "agx" ]; then
    echo "🔵 [Mode] 切換至 AGX 模式 (使用 .env.agx)"
    ENV_FILE=".env.agx"
    
    # 檢查檔案是否存在
    if [ ! -f "$ENV_FILE" ]; then
        echo "❌ 錯誤: 找不到 $ENV_FILE 檔案！"
        exit 1
    fi
else
    echo "💻 [Mode] 切換至 PC 模式 (使用 .env)"
    ENV_FILE=".env"
fi

# ==========================================
# 2. X11 / GUI 權限設定 (通用邏輯)
# ==========================================

# 只有在檢測到 DISPLAY 環境變數時才執行 (避免在純 SSH 無轉發環境報錯)
if [ -n "$DISPLAY" ]; then
    echo "🎨 [X11] 偵測到 DISPLAY=$DISPLAY，正在設定 Xauthority..."
    
    XAUTH=/tmp/.docker.xauth
    # 建立檔案 (如果不存在)
    touch $XAUTH

    # 製作 "萬用鑰匙"
    # 如果 xauth 指令失敗 (例如 WSL 有時會怪怪的)，不中斷腳本執行 (|| true)
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - || echo "⚠️  警告: xauth merge 遇到問題，嘗試繼續..."

    # 確保權限正確
    chmod 644 $XAUTH
else
    echo "🚫 [X11] 未偵測到 DISPLAY，跳過 GUI 權限設定 (Headless Mode)"
fi

# ==========================================
# 3. 啟動 Docker Compose
# ==========================================

echo "🚀 [Start] 正在啟動 Docker Compose..."
echo "📂 使用設定檔: $ENV_FILE"

# --env-file 參數告訴 Compose 讀取哪一個變數檔
# -p (project-name) 非必要，但推薦加上以避免名稱衝突，這裡用 agx_ros
docker compose --env-file "$ENV_FILE" -p agx_ros up -d

echo "✅ 啟動完成！"