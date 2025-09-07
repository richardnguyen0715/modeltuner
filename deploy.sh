#!/bin/bash
# filepath: deploy.sh

# =============================================================================
# Vietnamese VQA Model Training Deployment Script
# =============================================================================

set -e  # Exit on any error

# Configuration
SERVER_USER="your_username"
SERVER_HOST="your_gpu_server_ip"
SERVER_PORT="22"
REMOTE_DIR="/home/${SERVER_USER}/vqa_training"
LOCAL_PROJECT_DIR="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Function to check if SSH connection works
check_ssh_connection() {
    log "Checking SSH connection to ${SERVER_USER}@${SERVER_HOST}..."
    if ssh -p ${SERVER_PORT} -o ConnectTimeout=10 ${SERVER_USER}@${SERVER_HOST} "echo 'SSH connection successful'" > /dev/null 2>&1; then
        log "✅ SSH connection successful"
    else
        error "❌ Cannot connect to server. Please check your SSH credentials and server status."
    fi
}

# Function to create remote directory structure
create_remote_structure() {
    log "Creating remote directory structure..."
    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'EOF'
mkdir -p /home/$(whoami)/vqa_training/{bartphobeit,data/preprocessed_images,data/text,logs,checkpoints,results}
echo "Remote directory structure created successfully"
EOF
}

# Function to sync project files
sync_project_files() {
    log "Syncing project files to remote server..."
    
    # Sync bartphobeit code
    log "Uploading bartphobeit module..."
    rsync -avz -e "ssh -p ${SERVER_PORT}" \
        --progress \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        bartphobeit/ ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/bartphobeit/
    
    # Sync requirements.txt
    log "Uploading requirements.txt..."
    scp -P ${SERVER_PORT} requirements.txt ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/
    
    # Sync data files (with compression for large files)
    log "Uploading data files (this may take a while)..."
    rsync -avz -e "ssh -p ${SERVER_PORT}" \
        --progress \
        --compress \
        data/ ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/data/
    
    log "✅ All files synced successfully"
}

# Function to setup remote environment
setup_remote_environment() {
    log "Setting up remote Python environment..."
    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'EOF'
cd /home/$(whoami)/vqa_training

# Check Python version
python3 --version

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  nvidia-smi not found. GPU may not be available."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify key packages
echo "Verifying installations:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python3 -c "import torch; print(f'CUDA Device Count: {torch.cuda.device_count()}')"
    python3 -c "import torch; print(f'CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
fi
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo "✅ Remote environment setup completed"
EOF
}

# Function to create training script
create_training_script() {
    log "Creating remote training script..."
    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'EOF'
cd /home/$(whoami)/vqa_training

cat > run_training.sh << 'TRAINING_SCRIPT'
#!/bin/bash

# Vietnamese VQA Model Training Script
set -e

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/$(whoami)/vqa_training:$PYTHONPATH"

# Create timestamp for this training run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/training_${TIMESTAMP}"
mkdir -p ${LOG_DIR}

echo "🚀 Starting Vietnamese VQA Model Training"
echo "Timestamp: ${TIMESTAMP}"
echo "Log Directory: ${LOG_DIR}"
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Change to bartphobeit directory
cd bartphobeit

# Run training with comprehensive logging
echo "Starting training..." | tee ../${LOG_DIR}/training.log

python3 main.py 2>&1 | tee -a ../${LOG_DIR}/training.log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!" | tee -a ../${LOG_DIR}/training.log
    
    # Move results to timestamped directory
    if [ -f "best_vqa_model.pth" ]; then
        mv best_vqa_model.pth ../checkpoints/best_vqa_model_${TIMESTAMP}.pth
        echo "Model saved: checkpoints/best_vqa_model_${TIMESTAMP}.pth" | tee -a ../${LOG_DIR}/training.log
    fi
    
    if [ -f "best_fuzzy_model.pth" ]; then
        mv best_fuzzy_model.pth ../checkpoints/best_fuzzy_model_${TIMESTAMP}.pth
        echo "Fuzzy model saved: checkpoints/best_fuzzy_model_${TIMESTAMP}.pth" | tee -a ../${LOG_DIR}/training.log
    fi
    
    if [ -f "best_vqa_results.json" ]; then
        mv best_vqa_results.json ../results/best_vqa_results_${TIMESTAMP}.json
        echo "Results saved: results/best_vqa_results_${TIMESTAMP}.json" | tee -a ../${LOG_DIR}/training.log
    fi
    
    if [ -f "final_evaluation_results.json" ]; then
        mv final_evaluation_results.json ../results/final_evaluation_results_${TIMESTAMP}.json
        echo "Final results saved: results/final_evaluation_results_${TIMESTAMP}.json" | tee -a ../${LOG_DIR}/training.log
    fi
    
    echo "🎉 All files organized successfully!" | tee -a ../${LOG_DIR}/training.log
    
else
    echo "❌ Training failed with exit code $?" | tee -a ../${LOG_DIR}/training.log
    exit 1
fi
TRAINING_SCRIPT

chmod +x run_training.sh
echo "✅ Training script created: run_training.sh"
EOF
}

# Function to create monitoring script
create_monitoring_script() {
    log "Creating monitoring script..."
    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'EOF'
cd /home/$(whoami)/vqa_training

cat > monitor_training.sh << 'MONITOR_SCRIPT'
#!/bin/bash

# Training Monitor Script
echo "🔍 VQA Training Monitor"
echo "======================"

# Show GPU usage
echo ""
echo "📊 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F, '{printf "GPU: %s%% | Memory: %sMB/%sMB | Temp: %s°C\n", $1, $2, $3, $4}'

# Show latest log
if [ -d "logs" ]; then
    LATEST_LOG=$(find logs -name "training.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_LOG" ]; then
        echo ""
        echo "📝 Latest Training Log (last 20 lines):"
        echo "File: $LATEST_LOG"
        echo "----------------------------------------"
        tail -20 "$LATEST_LOG"
    fi
fi

# Show training processes
echo ""
echo "🔄 Training Processes:"
ps aux | grep -E "(python.*main.py|run_training)" | grep -v grep || echo "No training processes found"

# Show disk usage
echo ""
echo "💾 Disk Usage:"
df -h . | grep -v Filesystem
echo ""
du -sh checkpoints/ results/ logs/ 2>/dev/null | head -10
MONITOR_SCRIPT

chmod +x monitor_training.sh
echo "✅ Monitor script created: monitor_training.sh"
EOF
}

# Function to start training
start_training() {
    log "Starting training on remote server..."
    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'EOF'
cd /home/$(whoami)/vqa_training

echo "🚀 Launching training in tmux session..."

# Create or attach to tmux session
if tmux has-session -t vqa_training 2>/dev/null; then
    echo "Existing tmux session found. Killing old session..."
    tmux kill-session -t vqa_training
fi

# Start new tmux session with training
tmux new-session -d -s vqa_training './run_training.sh'

echo "✅ Training started in tmux session 'vqa_training'"
echo ""
echo "📋 Useful commands:"
echo "  - Monitor training: ./monitor_training.sh"
echo "  - Attach to session: tmux attach-session -t vqa_training"
echo "  - View logs: tail -f logs/training_*/training.log"
echo "  - Check GPU: nvidia-smi"
echo ""
echo "🔍 Current status:"
EOF

    # Show initial status
    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} 'cd /home/$(whoami)/vqa_training && ./monitor_training.sh'
}

# Function to download results
download_results() {
    log "Downloading training results..."
    
    # Create local results directory
    mkdir -p local_results
    
    # Download checkpoints
    log "Downloading model checkpoints..."
    rsync -avz -e "ssh -p ${SERVER_PORT}" \
        --progress \
        ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/checkpoints/ \
        local_results/checkpoints/
    
    # Download results
    log "Downloading result files..."
    rsync -avz -e "ssh -p ${SERVER_PORT}" \
        --progress \
        ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/results/ \
        local_results/results/
    
    # Download latest logs
    log "Downloading training logs..."
    rsync -avz -e "ssh -p ${SERVER_PORT}" \
        --progress \
        ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/logs/ \
        local_results/logs/
    
    log "✅ Results downloaded to local_results/"
}

# Main deployment function
main() {
    case "${1:-}" in
        "setup")
            log "🚀 Starting full deployment setup..."
            check_ssh_connection
            create_remote_structure
            sync_project_files
            setup_remote_environment
            create_training_script
            create_monitoring_script
            log "✅ Setup completed! Use './deploy.sh train' to start training."
            ;;
        
        "sync")
            log "📤 Syncing files only..."
            check_ssh_connection
            sync_project_files
            log "✅ Files synced successfully!"
            ;;
        
        "train")
            log "🚀 Starting training..."
            check_ssh_connection
            start_training
            ;;
        
        "monitor")
            log "🔍 Checking training status..."
            ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} 'cd /home/$(whoami)/vqa_training && ./monitor_training.sh'
            ;;
        
        "logs")
            log "📝 Showing latest logs..."
            ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} 'cd /home/$(whoami)/vqa_training && find logs -name "training.log" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2- | xargs tail -50'
            ;;
        
        "download")
            download_results
            ;;
        
        "ssh")
            log "🔗 Connecting to server..."
            ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "cd ${REMOTE_DIR} && bash"
            ;;
        
        "stop")
            log "⏹️  Stopping training..."
            ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} 'tmux kill-session -t vqa_training 2>/dev/null && echo "Training stopped" || echo "No training session found"'
            ;;
        
        *)
            echo "Vietnamese VQA Model Training Deployment"
            echo "========================================"
            echo ""
            echo "Usage: $0 {setup|sync|train|monitor|logs|download|ssh|stop}"
            echo ""
            echo "Commands:"
            echo "  setup    - Full deployment setup (first time)"
            echo "  sync     - Sync code files to server"
            echo "  train    - Start model training"
            echo "  monitor  - Check training status"
            echo "  logs     - Show latest training logs"
            echo "  download - Download results and checkpoints"
            echo "  ssh      - Connect to server"
            echo "  stop     - Stop training session"
            echo ""
            echo "Example workflow:"
            echo "  1. Edit SERVER_USER and SERVER_HOST in this script"
            echo "  2. ./deploy.sh setup"
            echo "  3. ./deploy.sh train"
            echo "  4. ./deploy.sh monitor"
            echo "  5. ./deploy.sh download"
            ;;
    esac
}

# Run main function with all arguments
main "$@"