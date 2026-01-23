# ==============================================================================
#  AGX ROS Control Interface
# ==============================================================================

# --- [Configuration] ---
PROJECT_NAME := agx_ros
service ?= control
TASK_CONTAINER := control

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# --- [Auto-Detection] ---
CURRENT_CONTEXT := $(shell docker context show)

ifndef MODE
    ifneq (,$(findstring agx,$(CURRENT_CONTEXT)))
        MODE := agx
        ENV_FILE := .env.agx
    else
        ifeq ($(shell uname -m), aarch64)
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

COMPOSE_CMD := docker compose --env-file $(ENV_FILE) -p $(PROJECT_NAME)
TASK_EXEC := docker exec -it $(TASK_CONTAINER) bash -ic

# --- [Targets] ---
.DEFAULT_GOAL := help
.PHONY: help build up down join logs ps clean run stop view check-service check-task-service

help: ## Show available commands
	@echo "AGX ROS Control Interface"
	@echo "   Container: $(TASK_CONTAINER)"
	@echo "   Context:   $(CURRENT_CONTEXT)"
	@echo "   Mode:      $(MODE)"
	@echo "------------------------------------------------"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "%-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-env:
	@if [ ! -f $(ENV_FILE) ]; then echo "[Error] Config file '$(ENV_FILE)' not found."; exit 1; fi

check-service:
	@if [ -z "$$(docker ps -q -f name=$(service))" ]; then echo "[Error] Service '$(service)' is not running. Run 'make up' first."; exit 1; fi

check-task-service:
	@if [ -z "$$(docker ps -q -f name=$(TASK_CONTAINER))" ]; then echo "[Error] Container '$(TASK_CONTAINER)' is not running. Run 'make up' first."; exit 1; fi

build: check-env ## Build docker images
	@echo "[Info] Building in $(MODE) mode..."
	@$(COMPOSE_CMD) build $(s)

up: check-env ## Start services in background
	@echo "[Info] Starting services..."
	@$(COMPOSE_CMD) up -d $(s)
	@echo "[Info] System started."

rebuild: check-env ## Force rebuild and recreate containers
	@echo "[Info] Rebuilding and Restarting services..."
	@$(COMPOSE_CMD) up -d --build --force-recreate $(s)
	@echo "[Info] Rebuild complete."

down: ## Stop and remove containers
	@echo "[Info] Stopping services..."
	@$(COMPOSE_CMD) down --remove-orphans $(s)
	@echo "[Info] Services stopped."

join: check-service ## Enter container shell
	@echo "[Info] Entering $(service)..."
	@docker exec -it $(service) bash

logs: ## Follow service logs
	@$(COMPOSE_CMD) logs -f

clean: ## Remove containers and images
	@$(COMPOSE_CMD) down --rmi local -v --remove-orphans

# ==============================================================================
# Task Management (Tmux Integration)
# ==============================================================================

run: check-task-service ## Launch ROS task (Background)
	@echo "=========================================="
	@echo " AGX Task Launcher"
	@echo " Target: $(TASK_CONTAINER)"
	@echo "=========================================="
	@echo "1) Keyboard Control (agx_keyboard)"
	@echo "2) Lidar Mapping    (agx_lidar)"
	@echo "3) HDL Localization (agx_loc)"
	@echo "4) Realsense Camera (agx_camera)"
	@echo "q) Quit"
	@echo "------------------------------------------"
	@read -p "Select task: " choice; \
	if [ "$$choice" = "q" ]; then exit 0; fi; \
	S_NAME=""; CMD_MAIN=""; CMD_SUB=""; TYPE=""; \
	case $$choice in \
		1) S_NAME="agx_keyboard"; CMD_MAIN="rosrun rosserial_python serial_node.py _port:=/dev/ttyUSB0"; CMD_SUB="rosrun six_wheels_teleop imu_teletop_0908"; TYPE="split";; \
		2) S_NAME="agx_lidar";    CMD_MAIN="roslaunch velodyne_pointcloud VLP16_points.launch"; TYPE="single";; \
		3) S_NAME="agx_loc";      CMD_MAIN="roslaunch hdl_localization hdl_localization.launch"; TYPE="single";; \
		4) S_NAME="agx_camera";   CMD_MAIN="roslaunch realsense2_camera rs_camera.launch"; CMD_SUB="rosrun imu_filter_madgwick imu_filter_node _use_mag:=false _remove_gravity_vector:=true _output_rate:=100.0 /imu/data_raw:=/camera/imu"; TYPE="split";; \
		*) echo "[Error] Invalid option"; exit 1;; \
	esac; \
	if [ -z "$$S_NAME" ]; then echo "[Error] Failed to set session name."; exit 1; fi; \
	if TMUX= tmux has-session -t $$S_NAME 2>/dev/null; then \
		echo "[Info] Task '$$S_NAME' is already running."; \
		echo "       Use 'make view' to monitor."; \
		exit 0; \
	fi; \
	echo "[Info] Launching task: $$S_NAME..."; \
	TMUX= tmux new-session -d -s $$S_NAME; \
	sleep 1; \
	tmux set -g mouse on; \
	tmux send-keys -t $$S_NAME:0 "$(TASK_EXEC) '$$CMD_MAIN'" C-m; \
	if [ "$$TYPE" = "split" ]; then \
		tmux split-window -h -t $$S_NAME:0; \
		sleep 0.5; \
		tmux send-keys -t $$S_NAME:0.1 "sleep 2" C-m; \
		tmux send-keys -t $$S_NAME:0.1 "$(TASK_EXEC) '$$CMD_SUB'" C-m; \
	fi; \
	echo "[Info] Task started in background."; \
	echo "       Use 'make view' to monitor logs."; \
	echo "       Use 'make stop' to terminate."

stop: ## Terminate running tasks
	@LIST=$$(tmux ls -F "#{session_name}" 2>/dev/null | grep "^agx_" || true); \
	if [ -z "$$LIST" ]; then echo "[Info] No active tasks running."; exit 0; fi; \
	echo "=========================================="; \
	echo " Select task to terminate"; \
	echo "=========================================="; \
	echo "$$LIST" | awk '{print NR ") " $$0}'; \
	echo "a) Terminate All"; \
	echo "q) Quit"; \
	echo "------------------------------------------"; \
	read -p "Enter number: " k_choice; \
	if [ "$$k_choice" = "q" ]; then exit 0; fi; \
	if [ "$$k_choice" = "a" ]; then echo "$$LIST" | xargs -n 1 tmux kill-session -t; echo "[Info] All tasks terminated."; exit 0; fi; \
	TARGET=$$(echo "$$LIST" | sed -n "$${k_choice}p"); \
	if [ -n "$$TARGET" ]; then tmux kill-session -t "$$TARGET"; echo "[Info] $$TARGET terminated."; else echo "[Error] Invalid number."; fi

view: ## Attach to task session
	@LIST=$$(tmux ls -F "#{session_name}" 2>/dev/null | grep "^agx_" || true); \
	if [ -z "$$LIST" ]; then echo "[Info] No active tasks."; exit 0; fi; \
	echo "=========================================="; \
	echo " Select task to view"; \
	echo "=========================================="; \
	echo "$$LIST" | awk '{print NR ") " $$0}'; \
	echo "q) Quit"; \
	echo "------------------------------------------"; \
	read -p "Enter number: " v_choice; \
	if [ "$$v_choice" = "q" ]; then exit 0; fi; \
	TARGET=$$(echo "$$LIST" | sed -n "$${v_choice}p"); \
	if [ -n "$$TARGET" ]; then \
		if [ -n "$$TMUX" ]; then \
			echo "[Warn] Already in Tmux. Use 'Ctrl+B, s' to switch."; \
		else \
			echo "[Info] Attaching to $$TARGET... (Press Ctrl+B, d to detach)"; \
			tmux attach-session -t "$$TARGET"; \
		fi; \
	else \
		echo "[Error] Invalid number."; \
	fi