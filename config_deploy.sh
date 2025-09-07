#!/bin/bash
# filepath: config_deploy.sh

# Configuration file for deployment
# Edit these values according to your server setup

# Server Configuration
export SERVER_USER="your_username"          # Your username on the GPU server
export SERVER_HOST="192.168.1.100"         # IP address of your GPU server
export SERVER_PORT="22"                     # SSH port (usually 22)

# Optional: SSH key configuration
# export SSH_KEY_PATH="/path/to/your/private/key"  # Uncomment if using SSH key

# Remote paths
export REMOTE_DIR="/home/${SERVER_USER}/vqa_training"

# Training configuration
export CUDA_DEVICE="0"                      # GPU device to use (0, 1, 2, etc.)

echo "Configuration loaded:"
echo "  Server: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}"
echo "  Remote directory: ${REMOTE_DIR}"
echo "  CUDA device: ${CUDA_DEVICE}"