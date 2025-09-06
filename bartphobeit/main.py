from bartphobeit.config import get_improved_config
from bartphobeit.model import ImprovedVietnameseVQAModel, normalize_vietnamese_answer
from bartphobeit.trainer import ImprovedVQATrainer
from bartphobeit.dataset import EnhancedVietnameseVQADataset
from bartphobeit.BARTphoBEIT import prepare_data_from_dataframe
from transformers import AutoTokenizer, ViTFeatureExtractor
import pandas as pd
from torch.utils.data import DataLoader
import torch
import warnings
warnings.filterwarnings('ignore')

def analyze_data_balance(questions):
    """Analyze answer distribution for balance"""
    from collections import Counter
    from bartphobeit.model import normalize_vietnamese_answer
    
    answers = [normalize_vietnamese_answer(q['ground_truth']) for q in questions]
    answer_counts = Counter(answers)
    
    print(f"\nData Balance Analysis:")
    print(f"  Total unique answers: {len(answer_counts)}")
    print(f"  Top 10 most common answers:")
    
    for answer, count in answer_counts.most_common(10):
        percentage = (count / len(answers)) * 100
        print(f"    '{answer}': {count} ({percentage:.2f}%)")
    
    # Check for severe imbalance
    most_common_count = answer_counts.most_common(1)[0][1]
    imbalance_ratio = most_common_count / len(answers)
    
    if imbalance_ratio > 0.3:
        print(f"  ⚠️ Severe imbalance detected: {imbalance_ratio:.2f} of answers are the same")
    else:
        print(f"  ✓ Data balance looks reasonable: {imbalance_ratio:.2f}")
    
    return answer_counts

def main():
    """Enhanced main training function"""
    
    # Load improved configuration
    config = get_improved_config()
    
    print(f"Enhanced Vietnamese VQA Training")
    print(f"Using device: {config['device']}")
    print(f"Configuration:")
    for key, value in config.items():
        if key != 'device':
            print(f"  {key}: {value}")
    
    # Load and prepare data
    print(f"\nLoading data...")
    df = pd.read_csv('/home/tgng/coding/BARTphoBEIT_imple/text/evaluate_60k_data_balanced.csv')
    questions = prepare_data_from_dataframe(df)
    
    # Analyze data balance
    analyze_data_balance(questions)
    
    # Split data
    split_idx = int(0.8 * len(questions))
    train_questions = questions[:split_idx]
    val_questions = questions[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train questions: {len(train_questions):,}")
    print(f"  Validation questions: {len(val_questions):,}")
    
    # Initialize tokenizers and feature extractor
    print(f"\nLoading tokenizers and feature extractor...")
    question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    answer_tokenizer = AutoTokenizer.from_pretrained(config['decoder_model'])
    feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
    
    # Test data normalization
    print(f"\nTesting answer normalization...")
    test_answers = ["Con chim đang bay.", "con chim", "Chim bay trong trời", "Có nhiều người đang đi bộ"]
    for ans in test_answers:
        normalized = normalize_vietnamese_answer(ans)
        print(f"  '{ans}' → '{normalized}'")
    
    # Create enhanced datasets
    print(f"\nCreating enhanced datasets with augmentation...")
    train_dataset = EnhancedVietnameseVQADataset(
        train_questions, config['image_dir'], question_tokenizer, 
        answer_tokenizer, feature_extractor, config['max_length'],
        use_augmentation=config.get('use_data_augmentation', False),
        augment_ratio=config.get('augment_ratio', 0.2)
    )
    train_dataset.set_training(True)  # Enable training mode
    
    val_dataset = EnhancedVietnameseVQADataset(
        val_questions, config['image_dir'], question_tokenizer, 
        answer_tokenizer, feature_extractor, config['max_length'],
        use_augmentation=False  # No augmentation for validation
    )
    val_dataset.set_training(False)  # Disable training mode
    
    # Create data loaders with reduced num_workers to avoid issues
    print(f"\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=0, pin_memory=True if config['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=0, pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Initialize enhanced model
    print(f"\nInitializing enhanced model...")
    model = ImprovedVietnameseVQAModel(config)
    model = model.to(config['device'])
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nEnhanced Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    
    # Test model forward pass
    print(f"\nTesting model forward pass...")
    try:
        test_batch = next(iter(train_loader))
        for key, value in test_batch.items():
            if isinstance(value, torch.Tensor):
                test_batch[key] = value.to(config['device'])
        
        with torch.no_grad():
            outputs = model(
                pixel_values=test_batch['pixel_values'][:2],  # Test with 2 samples
                question_input_ids=test_batch['question_input_ids'][:2],
                question_attention_mask=test_batch['question_attention_mask'][:2],
                answer_input_ids=test_batch['answer_input_ids'][:2],
                answer_attention_mask=test_batch['answer_attention_mask'][:2]
            )
            print(f"  ✓ Forward pass successful")
            print(f"  Loss: {outputs.loss.item():.4f}")
            print(f"  Logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"  ❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test inference mode
    print(f"\nTesting inference mode...")
    try:
        with torch.no_grad():
            generated_ids = model(
                pixel_values=test_batch['pixel_values'][:1],
                question_input_ids=test_batch['question_input_ids'][:1],
                question_attention_mask=test_batch['question_attention_mask'][:1]
            )
            
            pred_text = model.decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"  ✓ Inference successful")
            print(f"  Sample prediction: '{pred_text}'")
            print(f"  Sample ground truth: '{test_batch['answer_text'][0]}'")
    except Exception as e:
        print(f"  ❌ Error in inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize enhanced trainer
    print(f"\nInitializing enhanced trainer...")
    trainer = ImprovedVQATrainer(model, train_loader, val_loader, config['device'], config)
    
    # Start training
    print(f"\n{'='*80}")
    print(f"STARTING ENHANCED TRAINING")
    print(f"{'='*80}")
    print(f"Training for {config['num_epochs']} epochs with:")
    print(f"  Decoder LR: {config['decoder_lr']:.2e}")
    print(f"  Encoder LR: {config['encoder_lr']:.2e}")
    print(f"  Vision LR: {config['vision_lr']:.2e}")
    print(f"  Label smoothing: {config['label_smoothing']}")
    print(f"  Dropout rate: {config['dropout_rate']}")
    print(f"  Warmup ratio: {config.get('warmup_ratio', 0.1)}")
    print(f"  Data augmentation: {config.get('use_data_augmentation', False)}")
    print(f"  Wandb logging: {config.get('use_wandb', False)}")
    
    try:
        best_accuracy = trainer.train(config['num_epochs'])
        
        print(f"\n{'='*80}")
        print(f"🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print(f"{'='*80}")
        print(f"Best fuzzy accuracy achieved: {best_accuracy:.4f}")
        print(f"Model and checkpoints saved in current directory")
        print(f"Predictions saved for analysis")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        print(f"Saving current state...")
        trainer.save_checkpoint(trainer.global_step // len(train_loader), {}, is_best=False)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()