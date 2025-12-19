# ==============================================================================
#  AGX ROS Project Manager 
# ==============================================================================

# --- [Configuration] ---
PROJECT_NAME := agx_ros
# 強制開啟 BuildKit 加速
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# --- [Auto-Detection] ---
CURRENT_CONTEXT := $(shell docker context show)
# 預設服務 (make join 用)
service ?= isaac_ros

# 智慧判斷模式
ifndef MODE
	ifneq (,$(findstring agx,$(CURRENT_CONTEXT)))
		MODE := agx
		ENV_FILE := .env.agx
	else
		# 檢查是否為 ARM 架構
		ifeq ($(shell uname -m), aarch64)
			MODE := agx
			ENV_FILE := .env.agx
		else
			MODE := pc
			ENV_FILE := .env
		endif
	endif
else
	# 手動指定模式
	ifeq ($(MODE), agx)
		ENV_FILE := .env.agx
	else
		ENV_FILE := .env
	endif
endif

# 定義 Docker Compose 指令變數 (減少重複程式碼)
COMPOSE_CMD := docker compose --env-file $(ENV_FILE) -p $(PROJECT_NAME)

# --- [Targets] ---
.DEFAULT_GOAL := help
.PHONY: help build up down restart join logs ps clean shell-check

help: ## 顯示指令清單
	@echo "🤖 \033[1;34mAGX ROS Project Manager\033[0m"
	@echo "   Context: \033[1;35m$(CURRENT_CONTEXT)\033[0m"
	@echo "   Mode:    \033[1;33m$(MODE)\033[0m"
	@echo "   Env:     $(ENV_FILE)"
	@echo "------------------------------------------------"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "❌ Error: Config file '$(ENV_FILE)' not found!"; \
		exit 1; \
	fi

build: check-env ## 🛠️  建置映像檔 (Changed Only)
	@echo "🔨 Building in [\033[1;33m$(MODE)\033[0m] mode..."
	@$(COMPOSE_CMD) build

up: check-env ## 🚀 啟動系統 (Fast Boot)
	@echo "🚀 Starting services..."
	@$(COMPOSE_CMD) up -d
	@echo "✅ System is running. Use 'make logs' to monitor."

rebuild: check-env ## 🔄 強制重建並重啟
	@echo "🔄 Rebuilding and Restarting..."
	@$(COMPOSE_CMD) up -d --build --force-recreate

down: ## 🛑 停止系統
	@echo "🛑 Stopping services..."
	@$(COMPOSE_CMD) down --remove-orphans

join: ## 🐳 進入容器 (預設: isaac_ros)
	@echo "🐳 Entering \033[1;32m$(service)\033[0m..."
	@docker exec -it $(service) bash || echo "❌ Failed. Is '$(service)' running?"

logs: ## 📄 查看日誌 (Ctrl+C 離開)
	@$(COMPOSE_CMD) logs -f

ps: ## 📊 查看容器狀態
	@$(COMPOSE_CMD) ps

clean: ## 🧹 清理停止的容器與無用網路 (釋放空間)
	@echo "🧹 Cleaning up project resources..."
	@$(COMPOSE_CMD) down --rmi local -v --remove-orphans
	@echo "✨ Cleaned."

# --- [Advance: Shell Autocomplete Helper] ---
# 這段是給 Shell (Zsh/Bash) 用來做自動補全的，普通執行不會用到
_services:
	@$(COMPOSE_CMD) config --services 2>/dev/null