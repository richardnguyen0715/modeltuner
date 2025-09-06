import torch

def get_improved_config():
    """Improved configuration with all enhancements"""
    return {
        'vision_model': 'google/vit-base-patch16-224-in21k',
        'text_model': 'vinai/phobert-large',
        'decoder_model': 'vinai/bartpho-word',
        'hidden_dim': 1024,  # PhoBERT-large dimension
        'max_length': 128,
        'batch_size': 16,
        'num_epochs': 10,
        'image_dir': '/home/tgng/coding/modeltuner/data/preprocessed_images',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Staged training configuration
        'stage1_epochs': 2,  # Freeze encoders
        'stage2_epochs': 8,  # Partial unfreeze
        
        # Different learning rates
        'decoder_lr': 1e-4,
        'encoder_lr': 1e-5,
        'vision_lr': 5e-6,
        
        # Enhanced scheduler configuration
        'warmup_ratio': 0.1,  # 10% of total steps for warmup
        'scheduler_type': 'linear_decay_with_warmup',
        'weight_decay': 0.01,
        
        # Enhanced regularization
        'label_smoothing': 0.1,  # Use CrossEntropyLoss with label smoothing
        'dropout_rate': 0.2,
        
        # Unfreezing strategy
        'unfreeze_last_n_layers': 4,  # Unfreeze last 4 layers of text encoder
        
        # Data augmentation
        'use_data_augmentation': True,
        'augment_ratio': 0.2,  # 20% of data will be augmented
        
        # Logging and checkpoints
        'use_wandb': True,
        'project_name': 'Vietnamese-VQA-BARTPhoBEIT',
        'save_every_n_epochs': 1,  # Save checkpoint every epoch
        'keep_last_n_checkpoints': 5,
        
        # Enhanced evaluation
        'evaluate_every_n_steps': 5000,
        'save_predictions': True,
        'calculate_bleu_rouge': True,
        'calculate_cider': True,
    }