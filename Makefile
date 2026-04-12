# ==============================================================================
#  AGX ROS Control Interface (Unified Services)
# ==============================================================================

PROJECT_NAME   := agx_ros

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# --- [Auto-Detection] ---
CURRENT_CONTEXT := $(shell docker context show)
MODE := pc
ENV_FILE := .env
COMPOSE_FILE := docker-compose.pc.yaml

ifneq (,$(findstring agx,$(CURRENT_CONTEXT)))
    MODE := agx
    ENV_FILE := .env.agx
    COMPOSE_FILE := docker-compose.yaml
else ifeq ($(shell uname -m), aarch64)
    MODE := agx
    ENV_FILE := .env.agx
    COMPOSE_FILE := docker-compose.yaml
endif

COMPOSE_CMD := docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) -p $(PROJECT_NAME)

.DEFAULT_GOAL := help
.SILENT:
.PHONY: help build up down rebuild logs ps clean dashboard join

help:
	@echo "AGX ROS Control Interface [Mode: $(MODE)]"
	@echo "================================================"
	@echo "  make up [s=name]        Start service (auto-prompts if no s)"
	@echo "  make down [s=name]      Stop service (auto-prompts if no s)"
	@echo "  make build [s=name]     Build image (auto-prompts if no s)"
	@echo "  make rebuild [s=name]   Rebuild service (auto-prompts if no s)"
	@echo "  make logs [s=name]      View logs (auto-prompts if no s)"
	@echo "  make ps                 View container status"
	@echo "  make clean              Remove all containers & images"
	@echo "  make dashboard          Launch Web Dashboard (port 8080)"
	@echo "  make join [c=name]      Enter container (auto-prompts if no c)"

check-env:
	@if [ ! -f $(ENV_FILE) ]; then echo "[Error] Config file '$(ENV_FILE)' not found."; exit 1; fi

# --- Interactive Menu Script (Unified) ---
# Usage: bash -c "$$SCRIPT_MENU" _ <mode> <type>
#   <mode>: all (parse compose config) | running (parse docker ps)
#   <type>: multi (allows 'a' for all, default 'a') | single (exact 1 choice)
define SCRIPT_MENU
    MODE=$$1
    TYPE=$$2
    echo "==========================================" >&2
    if [ "$$MODE" = "running" ]; then
        echo " AGX Service Menu (Running)" >&2
        SVCS=$$($(COMPOSE_CMD) ps --services --filter "status=running" 2>/dev/null | sort)
        if [ -z "$$SVCS" ]; then echo "[Info] No running services found." >&2; exit 1; fi
    else
        echo " AGX Service Menu" >&2
        SVCS=$$($(COMPOSE_CMD) config --services 2>/dev/null | sort)
        if [ -z "$$SVCS" ]; then echo "[Error] Failed to parse services from $(COMPOSE_FILE)!" >&2; exit 1; fi
    fi
    echo "==========================================" >&2
    COUNT=0
    SERVICES=""
    for svc in $$SVCS; do
        COUNT=$$((COUNT + 1))
        echo "$$COUNT) $$svc" >&2
        if [ -z "$$SERVICES" ]; then SERVICES="$$svc"; else SERVICES="$$SERVICES $$svc"; fi
    done
    echo "------------------------------------------" >&2
    if [ "$$TYPE" = "single" ]; then
        echo "q) Quit" >&2
        echo "------------------------------------------" >&2
        read -p "Select a service [e.g. 1]: " INPUT < /dev/tty
    else
        echo "a) ALL         q) Quit" >&2
        echo "------------------------------------------" >&2
        read -p "Select service(s) [e.g. 1 3, default: a]: " INPUT < /dev/tty
    fi
    
    if [ "$$INPUT" = "q" ] || ([ "$$TYPE" = "single" ] && [ -z "$$INPUT" ]); then echo "quit"; exit 0; fi
    
    TARGETS=""
    if [ "$$TYPE" = "multi" ] && ([ -z "$$INPUT" ] || [ "$$INPUT" = "a" ]); then
        echo "all"
        exit 0
    fi
    
    CHOICES=$$(echo "$$INPUT" | tr ',' ' ')
    for c in $$CHOICES; do
        T=$$(echo "$$SERVICES" | awk -v i="$$c" '{print $$i}')
        if [ -n "$$T" ]; then TARGETS="$$TARGETS $$T"; fi
        if [ "$$TYPE" = "single" ] && [ -n "$$TARGETS" ]; then break; fi
    done
    echo "$$TARGETS"
endef
export SCRIPT_MENU

build: check-env
	@if [ -n "$(s)" ]; then \
		echo "🚀 Building: $(s)"; $(COMPOSE_CMD) build $(s); \
	else \
		RES=$$(bash -c "$$SCRIPT_MENU" _ "all" "multi"); \
		if [ "$$RES" = "quit" ]; then exit 0; \
		elif [ "$$RES" = "all" ]; then echo "🚀 Building ALL"; $(COMPOSE_CMD) build; \
		else echo "🚀 Building: $$RES"; $(COMPOSE_CMD) build $$RES; fi \
	fi

up: check-env
	@if [ -n "$(s)" ]; then \
		echo "🚀 Starting: $(s)"; $(COMPOSE_CMD) up -d $(s); \
	else \
		RES=$$(bash -c "$$SCRIPT_MENU" _ "all" "multi"); \
		if [ "$$RES" = "quit" ]; then exit 0; \
		elif [ "$$RES" = "all" ]; then echo "🚀 Starting ALL"; $(COMPOSE_CMD) up -d; \
		else echo "🚀 Starting: $$RES"; $(COMPOSE_CMD) up -d $$RES; fi \
	fi

rebuild: check-env
	@if [ -n "$(s)" ]; then \
		echo "🚀 Rebuilding: $(s)"; $(COMPOSE_CMD) up -d --build --force-recreate $(s); \
	else \
		RES=$$(bash -c "$$SCRIPT_MENU" _ "all" "multi"); \
		if [ "$$RES" = "quit" ]; then exit 0; \
		elif [ "$$RES" = "all" ]; then echo "🚀 Rebuilding ALL"; $(COMPOSE_CMD) up -d --build --force-recreate; \
		else echo "🚀 Rebuilding: $$RES"; $(COMPOSE_CMD) up -d --build --force-recreate $$RES; fi \
	fi

down:
	@if [ -n "$(s)" ]; then \
		echo "🚀 Stopping (rm -s -v -f): $(s)"; $(COMPOSE_CMD) rm -s -v -f $(s); \
	else \
		RES=$$(bash -c "$$SCRIPT_MENU" _ "running" "multi"); \
		if [ "$$RES" = "quit" ] || [ -z "$$RES" ]; then exit 0; \
		elif [ "$$RES" = "all" ]; then echo "🚀 Stopping ALL"; $(COMPOSE_CMD) down --remove-orphans; \
		else echo "🚀 Stopping: $$RES"; $(COMPOSE_CMD) rm -s -v -f $$RES; fi \
	fi

logs:
	@if [ -n "$(s)" ]; then \
		$(COMPOSE_CMD) logs -f $(s); \
	else \
		RES=$$(bash -c "$$SCRIPT_MENU" _ "all" "multi"); \
		if [ "$$RES" = "quit" ]; then exit 0; \
		elif [ "$$RES" = "all" ]; then $(COMPOSE_CMD) logs -f; \
		else $(COMPOSE_CMD) logs -f $$RES; fi \
	fi

ps:
	$(COMPOSE_CMD) ps

clean:
	$(COMPOSE_CMD) down --rmi local -v --remove-orphans

dashboard:
	docker compose --env-file $(ENV_FILE) -f dashboard/docker-compose.yaml -p $(PROJECT_NAME) up -d --build
	@echo "🚀 Dashboard: http://localhost:8080"

join: check-env
	@if [ -n "$(c)" ]; then \
		echo "🚀 Joining: $(c)"; $(COMPOSE_CMD) exec $(c) bash; \
	else \
		RES=$$(bash -c "$$SCRIPT_MENU" _ "running" "single"); \
		if [ "$$RES" = "quit" ] || [ -z "$$RES" ]; then exit 0; \
		else echo "🚀 Joining: $$RES"; $(COMPOSE_CMD) exec $$RES bash; fi \
	fi