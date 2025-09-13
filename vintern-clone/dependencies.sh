# Tạo virtual environment
python -m venv vintern_env
source vintern_env/bin/activate
pip install torch torchvision torchaudio
pip install transformers
pip install accelerate
pip install datasets
pip install Pillow
pip install peft
pip install deepspeed
pip install wandb