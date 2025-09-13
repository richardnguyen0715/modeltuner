# configs/training_config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model settings
    use_lora: bool = True
    
    # Data settings
    train_data_path: str = "data/processed/train.json"
    val_data_path: str = "data/processed/val.json"
    image_root: str = "data/images/"
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 32  # Effective batch size = 128
    learning_rate: float = 4e-5
    num_epochs: int = 3
    max_length: int = 4096
    warmup_steps: int = 100
    
    # Hardware settings
    num_gpus: int = 4
    mixed_precision: str = "fp16"
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 1000
    output_dir: str = "models/checkpoints"
    
    # Logging
    logging_steps: int = 100
    run_name: str = "vintern-1b-custom"