import torch

def get_improved_config():
    """Enhanced config with resume training support"""
    config = {
        # Model architecture
        'vision_model': 'google/vit-base-patch16-224-in21k',
        'text_model': 'vinai/bartpho-syllable',
        'decoder_model': 'vinai/bartpho-syllable',
        
        # Training parameters
        'batch_size': 8,
        'num_epochs': 20,
        'stage1_epochs': 8,
        'unfreeze_last_n_layers': 2,
        
        # Learning rates
        'decoder_lr': 5e-5,
        'encoder_lr': 2e-5,
        'vision_lr': 1e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        
        # Data settings
        'image_dir': '/home/tgng/coding/modeltuner/data/preprocessed_images/',
        'max_length': 128,
        'use_data_augmentation': True,
        'augment_ratio': 0.2,
        
        # Enhanced BARTPhoBEiT features
        'hidden_dim': 1024,
        'num_multiway_layers': 6,
        'dropout_rate': 0.1,
        'label_smoothing': 0.1,
        
        # VQ-KD Visual Tokenizer
        'use_vqkd': True,
        'visual_vocab_size': 8192,
        
        # Block-wise vision masking
        'use_unified_masking': True,
        'vision_mask_ratio': 0.4,
        'vision_mask_block_size': 4,
        'text_mask_ratio': 0.15,
        'multimodal_text_mask_ratio': 0.5,
        
        # WUPS metrics
        'calculate_wups': True,
        
        # Enhanced evaluation
        'pool_target_length': 32,
        'evaluate_every_n_steps': 0,  # Set to 0 to disable step evaluation
        
        # Logging and saving
        'use_wandb': False,
        'project_name': 'BARTPhoBEiT-Vietnamese-VQA',
        'save_every_n_epochs': 1,
        'keep_last_n_checkpoints': 5,
        'save_predictions': True,
        
        # ✨ NEW: Resume training configuration
        'resume_training': False,  # Enable/disable resume functionality
        'resume_from_checkpoint': None,  # Path to specific checkpoint or 'latest' for auto-detect
        'auto_resume': True,  # Automatically find and resume from latest checkpoint
        'resume_strict': True,  # Strict mode for checkpoint loading
        'reset_optimizer_on_resume': False,  # Reset optimizer state when resuming
        'reset_scheduler_on_resume': False,  # Reset scheduler state when resuming
        'resume_best_scores': True,  # Restore best scores from checkpoint
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    return config