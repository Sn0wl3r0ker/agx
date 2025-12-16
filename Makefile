# ==============================================================================
#  AGX ROS Project Manager (Fast Boot Version)
# ==============================================================================

# 強制開啟 BuildKit (加速建置時使用)
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# 1. 自動偵測 Context 與 Mode
CURRENT_CONTEXT := $(shell docker context show)
ifndef MODE
	ifneq (,$(findstring agx,$(CURRENT_CONTEXT)))
		MODE := agx
		ENV_FILE := .env.agx
	else
		ARCH := $(shell uname -m)
		ifeq ($(ARCH), aarch64)
			MODE := agx
			ENV_FILE := .env.agx
		else
			MODE := pc
			ENV_FILE := .env
		endif
	endif
else
	ifeq ($(MODE), agx)
		ENV_FILE := .env.agx
	else
		ENV_FILE := .env
	endif
endif

PROJECT_NAME := agx_ros
service ?= isaac_ros

.DEFAULT_GOAL := help
.PHONY: help build up down join logs ps check-env

help: ## 顯示指令清單
	@echo "🤖 \033[1;34mAGX ROS Project Manager\033[0m"
	@echo "   Context: \033[1;35m$(CURRENT_CONTEXT)\033[0m"
	@echo "   Mode:    \033[1;33m$(MODE)\033[0m"
	@echo "   Env:     $(ENV_FILE)"
	@echo "------------------------------------------------"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "❌ Error: 找不到設定檔 '$(ENV_FILE)'"; \
		exit 1; \
	fi

build: check-env ## 🛠️  手動建置/更新 Docker 映像檔
	@echo "🔨 Building images in [\033[1;33m$(MODE)\033[0m] mode..."
	@CMD="docker compose --env-file $(ENV_FILE) -p $(PROJECT_NAME) build"; \
	echo "👉 Executing: $$CMD"; \
	$$CMD

up: check-env ## 🚀 啟動系統 (不重新建置，秒開)
	@echo "🚀 Starting services in [\033[1;33m$(MODE)\033[0m] mode..."
	@CMD="docker compose --env-file $(ENV_FILE) -p $(PROJECT_NAME) up -d"; \
	echo "👉 Executing: $$CMD"; \
	$$CMD

rebuild: check-env ## 🔄 強制重建並啟動 (等於 build + up)
	@echo "🔄 Rebuilding and starting..."
	@CMD="docker compose --env-file $(ENV_FILE) -p $(PROJECT_NAME) up -d --build"; \
	echo "👉 Executing: $$CMD"; \
	$$CMD

down: ## 🛑 關閉系統
	@echo "🛑 Stopping services..."
	@CMD="docker compose --env-file $(ENV_FILE) -p $(PROJECT_NAME) down --remove-orphans"; \
	echo "👉 Executing: $$CMD"; \
	$$CMD

join: ## 🐳 進入容器
	@echo "🐳 Entering container: $(service)..."
	@docker exec -it $(service) bash || echo "❌ 無法進入 $(service)，請確認它是否正在執行。"

logs: ## 📄 查看日誌
	@docker compose --env-file $(ENV_FILE) -p $(PROJECT_NAME) logs -f

ps: ## 📊 查看狀態
	@docker compose --env-file $(ENV_FILE) -p $(PROJECT_NAME) ps