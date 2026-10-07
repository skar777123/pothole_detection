#!/bin/bash
# ------------------------------------------------------------------------------
# Pothole Detection System
# Single-command launcher for backend API and ReactJS UI.
#
# Usage:
#   ./start.sh          : Activate venv (or create), setup, and start server
#   ./start.sh --setup  : Force reinstall of dependencies
#   ./start.sh --dev    : Run Vite dev server on port 3000 alongside backend
# ------------------------------------------------------------------------------

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8000

# Terminal styles
B="\033[1m"; G="\033[92m"; Y="\033[93m"; R="\033[91m"; C="\033[96m"; X="\033[0m"
info()  { echo -e "${G}${B}[OK]${X} $*"; }
warn()  { echo -e "${Y}${B}[WARN]${X} $*"; }
error() { echo -e "${R}${B}[ERROR]${X} $*"; exit 1; }
head()  { echo -e "\n${C}${B}=== $* ===${X}"; }

# Banner
if [ -t 1 ]; then clear 2>/dev/null || true; fi
echo -e "${C}${B}"
echo "  +--------------------------------------------+"
echo "  |   TF02-Pro LiDAR Pothole Detection System  |"
echo "  |   Unified FastAPI + ReactJS Application    |"
echo "  +--------------------------------------------+"
echo -e "${X}"

# Parse flags
DEV_MODE=false
FORCE_SETUP=false
for arg in "$@"; do
    case $arg in
        --dev)     DEV_MODE=true ;;
        --setup)   FORCE_SETUP=true ;;
    esac
done

# Step 1: Detect, create, and switch into Virtual Environment
head "Virtual Environment"

# Locate .venv or venv directory
if [ -d "$PROJECT_DIR/.venv" ]; then
    VENV_DIR="$PROJECT_DIR/.venv"
elif [ -d "$PROJECT_DIR/venv" ]; then
    VENV_DIR="$PROJECT_DIR/venv"
else
    VENV_DIR="$PROJECT_DIR/.venv"
fi

# Detect bootstrap python if venv does not exist yet
BOOTSTRAP_PY="python3"
if ! command -v "$BOOTSTRAP_PY" >/dev/null 2>&1; then
    BOOTSTRAP_PY="python"
fi

if [ ! -d "$VENV_DIR" ]; then
    warn "No virtual environment detected. Creating: $VENV_DIR"
    "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
    info "Created virtual environment at $VENV_DIR"
fi

# Find activate script and binaries (supports Linux, macOS, and Windows Git Bash)
ACTIVATE_SCRIPT=""
if [ -f "$VENV_DIR/bin/activate" ]; then
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
    VENV_BIN="$VENV_DIR/bin"
    PY="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
    VENV_BIN="$VENV_DIR/Scripts"
    PY="$VENV_DIR/Scripts/python"
    PIP="$VENV_DIR/Scripts/pip"
else
    error "Cannot find activate script in $VENV_DIR"
fi

# Switch environment if running in normal terminal
if [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
    if [ -n "$ACTIVATE_SCRIPT" ] && [ -f "$ACTIVATE_SCRIPT" ]; then
        # shellcheck disable=SC1090
        source "$ACTIVATE_SCRIPT"
        info "Switched terminal session to virtual environment: $VENV_DIR"
    else
        export PATH="$VENV_BIN:$PATH"
        export VIRTUAL_ENV="$VENV_DIR"
        info "Updated PATH to virtual environment: $VENV_BIN"
    fi
else
    info "Already running inside virtual environment: $VENV_DIR"
fi

# Step 2: Install or verify dependencies
do_setup() {
    head "Setup Dependencies"

    info "Upgrading pip in virtual environment..."
    "$PIP" install --upgrade pip --quiet || true

    info "Installing Python dependencies from requirements.txt..."
    "$PIP" install -r "$PROJECT_DIR/requirements.txt" --quiet

    if command -v id >/dev/null 2>&1; then
        if ! id -nG "$USER" 2>/dev/null | grep -qw dialout; then
            warn "Note: On Linux/Raspberry Pi, add your user to 'dialout' group for serial port access:"
            warn "  sudo usermod -aG dialout $USER"
        fi
    fi

    if [ -d "$PROJECT_DIR/frontend" ]; then
        cd "$PROJECT_DIR/frontend"
        if [ ! -d "node_modules" ]; then
            info "Installing frontend npm packages..."
            npm install --quiet
        fi
        info "Building production React frontend..."
        npm run build
        cd "$PROJECT_DIR"
    fi

    info "Setup completed successfully."
}

# Auto-run setup if dependencies are missing or if forced
if [ "$FORCE_SETUP" = true ]; then
    do_setup
elif ! "$PY" -c "import fastapi, serial, uvicorn, sklearn, websockets" >/dev/null 2>&1; then
    warn "Required Python modules not found in virtual environment. Running setup..."
    do_setup
fi

# Step 3: Check ML model
head "Checking ML Model"
if [ ! -f "$PROJECT_DIR/pothole_model.pkl" ]; then
    warn "ML model not found. Training pothole_model.pkl..."
    cd "$PROJECT_DIR"
    "$PY" model_train.py
    info "ML model trained successfully."
else
    info "ML model found (pothole_model.pkl)."
fi

# Step 4: Ensure React production build exists
if [ ! -d "$PROJECT_DIR/frontend/dist" ]; then
    warn "Frontend dist build not found. Building now..."
    cd "$PROJECT_DIR/frontend"
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    npm run build
    cd "$PROJECT_DIR"
    info "Frontend build ready."
fi

# Step 5: Clean shutdown handler
PIDS=()
cleanup() {
    echo ""
    warn "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    info "All services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Step 6: Detect LiDAR Hardware
head "Detecting LiDAR Hardware"
echo "Scanning for LiDAR on USB COM ports..."
DETECTED_PORT=$("$PY" -c "
import sys
try:
    from lidar_driver import auto_detect_lidar
    port = auto_detect_lidar(send_init=False)
    if port:
        print(port)
    else:
        print('NONE')
except Exception as e:
    print('NONE')
")

if [[ "$DETECTED_PORT" != "NONE" ]] && [[ -n "$DETECTED_PORT" ]]; then
    info "LiDAR successfully detected on: $DETECTED_PORT"
else
    warn "LiDAR not found. Ensure it is plugged in and drivers are installed."
fi

# Step 7: Start Server
head "Starting Application"

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")

echo ""
echo -e "  ${G}${B}Application is available at:${X}"
echo -e "  ${C}${B}  > http://${HOST_IP}:${PORT}${X}  (Network)"
echo -e "  ${C}${B}  > http://localhost:${PORT}${X} (Local)"
echo ""
echo -e "  ${Y}Press Ctrl+C to stop.${X}"
echo ""

cd "$PROJECT_DIR"
"$PY" server.py &
PIDS+=($!)

if [ "$DEV_MODE" = true ]; then
    info "Starting Vite dev server on port 3000..."
    cd "$PROJECT_DIR/frontend"
    npm run dev &
    PIDS+=($!)
    cd "$PROJECT_DIR"
fi

wait
