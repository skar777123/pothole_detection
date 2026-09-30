#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# pushToPi.sh
# Push local code changes directly to Raspberry Pi 4B over SSH / rsync / scp.
#
# Usage:
#   ./pushToPi.sh                       # Push to default IP (10.102.135.41)
#   ./pushToPi.sh 192.168.1.100         # Push to custom Pi IP
#   ./pushToPi.sh --pull                # Instruct Pi to git pull origin main
#   ./pushToPi.sh --run                 # Sync files and restart application
# ------------------------------------------------------------------------------

set -e

# Configuration
PI_USER="${PI_USER:-pi}"
PI_HOST="${1:-10.102.135.64}"
PI_DEST="/home/$PI_USER/pothole_detection"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Adjust if first arg was a flag
ACTION="sync"
for arg in "$@"; do
    case "$arg" in
        --pull) ACTION="pull" ;;
        --run)  ACTION="run" ;;
        --setup) ACTION="setup" ;;
        -h|--help)
            echo "Usage: $0 [PI_IP] [--pull|--run|--setup]"
            echo ""
            echo "Options:"
            echo "  PI_IP     Target Raspberry Pi IP address (default: 10.102.135.41)"
            echo "  --pull    Run 'git pull origin main' on the Pi instead of direct file sync"
            echo "  --run     Sync files and start ./start.sh on the Pi"
            echo "  --setup   Sync files and run ./start.sh --setup on the Pi"
            exit 0
            ;;
        *)
            if [[ "$arg" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                PI_HOST="$arg"
            fi
            ;;
    esac
done

# Styles
B="\033[1m"; G="\033[92m"; Y="\033[93m"; R="\033[91m"; C="\033[96m"; X="\033[0m"
info()  { echo -e "${G}${B}[OK]${X} $*"; }
warn()  { echo -e "${Y}${B}[WARN]${X} $*"; }
error() { echo -e "${R}${B}[ERROR]${X} $*"; exit 1; }
head()  { echo -e "\n${C}${B}=== $* ===${X}"; }

clear
echo -e "${C}${B}"
echo "  +--------------------------------------------------+"
echo "  |   Pothole Detection - Push to Raspberry Pi 4B    |"
echo "  +--------------------------------------------------+"
echo -e "${X}"
echo -e "  Target: ${B}$PI_USER@$PI_HOST:$PI_DEST${X}"
echo -e "  Key:    ${B}$SSH_KEY${X}"
echo ""

# Check SSH connection
head "1. Checking Pi Connectivity"
SSH_OPTS=(-o ConnectTimeout=5 -o StrictHostKeyChecking=no)
if [ -f "$SSH_KEY" ]; then
    SSH_OPTS+=(-i "$SSH_KEY")
fi

if ! ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "echo Connected" >/dev/null 2>&1; then
    error "Cannot connect to $PI_USER@$PI_HOST over SSH.\nMake sure the Raspberry Pi is turned on and on the same network."
fi
info "SSH connection to Raspberry Pi confirmed."

# Ensure destination directory exists on Pi and user has ownership
head "2. Preparing Remote Directory"
ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "mkdir -p $PI_DEST && sudo chown -R $PI_USER:$PI_USER $PI_DEST"
info "Remote directory verified at $PI_DEST"

# Mode: Git Pull
if [ "$ACTION" = "pull" ]; then
    head "3. Running Git Pull on Raspberry Pi"
    ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "cd $PI_DEST && git pull origin main"
    info "Raspberry Pi repository updated from GitHub."
    exit 0
fi

# Mode: Direct File Sync
head "3. Syncing Project Files"

# Rebuild frontend locally if node is installed
if [ -d "$PROJECT_DIR/frontend" ]; then
    info "Building production frontend assets locally..."
    (cd "$PROJECT_DIR/frontend" && npm run build --silent || npm run build)
fi

if command -v rsync >/dev/null 2>&1; then
    info "Using rsync for differential file sync..."
    SSH_CMD="ssh"
    if [ -f "$SSH_KEY" ]; then
        SSH_CMD="ssh -i $SSH_KEY"
    fi

    rsync -avz --delete \
        -e "$SSH_CMD" \
        --exclude '.git' \
        --exclude 'node_modules' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.venv' \
        --exclude 'venv' \
        --exclude 'cache' \
        "$PROJECT_DIR/" "$PI_USER@$PI_HOST:$PI_DEST/"
else
    info "rsync not found, using scp to transfer files..."
    SCP_OPTS=(-r -o StrictHostKeyChecking=no)
    if [ -f "$SSH_KEY" ]; then
        SCP_OPTS+=(-i "$SSH_KEY")
    fi

    # Core python files
    scp "${SCP_OPTS[@]}" "$PROJECT_DIR/server.py" "$PI_USER@$PI_HOST:$PI_DEST/"
    scp "${SCP_OPTS[@]}" "$PROJECT_DIR/lidar_driver.py" "$PI_USER@$PI_HOST:$PI_DEST/"
    scp "${SCP_OPTS[@]}" "$PROJECT_DIR/model_train.py" "$PI_USER@$PI_HOST:$PI_DEST/"
    scp "${SCP_OPTS[@]}" "$PROJECT_DIR/requirements.txt" "$PI_USER@$PI_HOST:$PI_DEST/"
    scp "${SCP_OPTS[@]}" "$PROJECT_DIR/start.sh" "$PI_USER@$PI_HOST:$PI_DEST/"
    if [ -f "$PROJECT_DIR/pothole_model.pkl" ]; then
        scp "${SCP_OPTS[@]}" "$PROJECT_DIR/pothole_model.pkl" "$PI_USER@$PI_HOST:$PI_DEST/"
    fi

    # Frontend files
    ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "mkdir -p $PI_DEST/frontend/dist"
    scp "${SCP_OPTS[@]}" "$PROJECT_DIR/frontend/package.json" "$PI_USER@$PI_HOST:$PI_DEST/frontend/"
    scp "${SCP_OPTS[@]}" "$PROJECT_DIR/frontend/dist/"* "$PI_USER@$PI_HOST:$PI_DEST/frontend/dist/"
fi

# Set executable permissions and fix line endings on Pi
head "4. Setting Permissions"
ssh "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "sed -i 's/\r$//' $PI_DEST/*.sh 2>/dev/null || true && chmod +x $PI_DEST/*.sh"
info "start.sh converted to Unix LF and marked executable on Raspberry Pi."

# Post actions
if [ "$ACTION" = "setup" ]; then
    head "5. Running Setup on Pi"
    ssh -t "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "cd $PI_DEST && ./start.sh --setup"
elif [ "$ACTION" = "run" ]; then
    head "5. Launching Application on Pi"
    ssh -t "${SSH_OPTS[@]}" "$PI_USER@$PI_HOST" "cd $PI_DEST && ./start.sh"
else
    echo ""
    echo -e "  ${G}${B}Sync complete!${X}"
    echo -e "  To start the application on the Pi, run:"
    echo -e "  ${C}ssh $PI_USER@$PI_HOST 'cd $PI_DEST && ./start.sh'${X}"
    echo ""
fi
