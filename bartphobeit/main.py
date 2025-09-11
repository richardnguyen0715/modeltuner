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
import os
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def validate_full_bart_config(config):
    """Validate Full BART configuration"""
    print(f"\n{'='*60}")
    print(f"VALIDATING FULL BART CONFIGURATION")
    print(f"{'='*60}")
    
    # Check BART model paths
    bart_models = ['vinai/bartpho-word', 'vinai/bartpho-syllable']
    
    text_model = config['text_model']
    decoder_model = config['decoder_model']
    
    if text_model not in bart_models:
        print(f"⚠️  Warning: text_model '{text_model}' is not a recognized BARTPho model")
        print(f"   Recommended: {bart_models}")
    else:
        print(f"✅ Text model: {text_model}")
    
    if decoder_model not in bart_models:
        print(f"⚠️  Warning: decoder_model '{decoder_model}' is not a recognized BARTPho model")
        print(f"   Recommended: {bart_models}")
    else:
        print(f"✅ Decoder model: {decoder_model}")
    
    if text_model != decoder_model:
        print(f"⚠️  Warning: Using different models for encoder and decoder")
        print(f"   For Full BART, both should be the same model")
    else:
        print(f"✅ Full BART: Using same model for encoder and decoder")
    
    # Validate BARTPhoBEiT enhancements
    print(f"\nBARTPhoBEiT Enhancements:")
    print(f"✅ VQ-KD Visual Tokenizer: {config.get('use_vqkd', False)}")
    print(f"✅ Block-wise vision masking: {config.get('use_unified_masking', False)} (ratio: {config.get('vision_mask_ratio', 0.4)})")
    print(f"✅ Multiway Transformers: {config.get('num_multiway_layers', 6)} layers")
    print(f"✅ WUPS metrics: {config.get('calculate_wups', False)}")
    print(f"✅ Full BART dimension: {config.get('hidden_dim', 1024)}")
    
    return config

def test_full_bart_tokenizers(config):
    """Test Full BART tokenizers compatibility"""
    print(f"\n{'='*60}")
    print(f"TESTING FULL BART TOKENIZERS")
    print(f"{'='*60}")
    
    try:
        # Load tokenizers
        question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
        answer_tokenizer = AutoTokenizer.from_pretrained(config['decoder_model'])
        
        print(f"✅ Question tokenizer loaded: {config['text_model']}")
        print(f"   Vocab size: {question_tokenizer.vocab_size:,}")
        print(f"   Special tokens: pad={question_tokenizer.pad_token_id}, "
              f"eos={question_tokenizer.eos_token_id}, "
              f"bos={question_tokenizer.bos_token_id}")
        
        print(f"✅ Answer tokenizer loaded: {config['decoder_model']}")
        print(f"   Vocab size: {answer_tokenizer.vocab_size:,}")
        print(f"   Special tokens: pad={answer_tokenizer.pad_token_id}, "
              f"eos={answer_tokenizer.eos_token_id}, "
              f"bos={answer_tokenizer.bos_token_id}")
        
        # Test Vietnamese tokenization
        test_question = "Có bao nhiều người trong hình?"
        test_answer = "hai người"
        
        q_tokens = question_tokenizer.encode(test_question)
        a_tokens = answer_tokenizer.encode(test_answer)
        
        print(f"\nTokenization test:")
        print(f"  Question: '{test_question}' -> {len(q_tokens)} tokens")
        print(f"  Answer: '{test_answer}' -> {len(a_tokens)} tokens")
        print(f"  Question tokens: {q_tokens}")
        print(f"  Answer tokens: {a_tokens}")
        
        # Decode test
        decoded_q = question_tokenizer.decode(q_tokens, skip_special_tokens=True)
        decoded_a = answer_tokenizer.decode(a_tokens, skip_special_tokens=True)
        
        print(f"  Decoded question: '{decoded_q}'")
        print(f"  Decoded answer: '{decoded_a}'")
        
        # Check compatibility
        if question_tokenizer.vocab_size == answer_tokenizer.vocab_size:
            print(f"✅ Tokenizers are compatible (same vocab size)")
        else:
            print(f"⚠️  Tokenizer vocab sizes differ: {question_tokenizer.vocab_size} vs {answer_tokenizer.vocab_size}")
        
        return question_tokenizer, answer_tokenizer
        
    except Exception as e:
        print(f"❌ Error loading tokenizers: {e}")
        raise e

def analyze_data_balance_with_wups_preview(questions):
    """Analyze data balance and preview WUPS evaluation"""
    from collections import Counter
    from bartphobeit.model import normalize_vietnamese_answer, compute_wups
    
    print(f"\n{'='*60}")
    print(f"DATA ANALYSIS WITH WUPS PREVIEW")
    print(f"{'='*60}")
    
    # Collect all answers (including all 5 per question)
    all_answers = []
    sample_questions_for_wups = []
    
    for i, q in enumerate(questions):
        if 'all_correct_answers' in q and q['all_correct_answers']:
            # Add all 5 correct answers
            normalized_answers = [normalize_vietnamese_answer(ans) for ans in q['all_correct_answers']]
            all_answers.extend(normalized_answers)
            
            # Collect samples for WUPS preview
            if i < 3:  # First 3 questions for preview
                sample_questions_for_wups.append({
                    'question': q['question'],
                    'answers': q['all_correct_answers'],
                    'normalized_answers': normalized_answers
                })
        else:
            # Fallback to ground_truth
            normalized = normalize_vietnamese_answer(q['ground_truth'])
            all_answers.append(normalized)
    
    answer_counts = Counter(all_answers)
    
    print(f"Dataset Statistics:")
    print(f"  Total questions: {len(questions):,}")
    print(f"  Total answer instances: {len(all_answers):,}")
    print(f"  Average answers per question: {len(all_answers) / len(questions):.2f}")
    print(f"  Unique normalized answers: {len(answer_counts):,}")
    
    print(f"\nTop 10 most common normalized answers:")
    for answer, count in answer_counts.most_common(10):
        percentage = (count / len(all_answers)) * 100
        print(f"    '{answer}': {count:,} ({percentage:.2f}%)")
    
    # Check for severe imbalance
    most_common_count = answer_counts.most_common(1)[0][1]
    imbalance_ratio = most_common_count / len(all_answers)
    
    if imbalance_ratio > 0.2:  # Lower threshold for multiple answers
        print(f"⚠️  Severe imbalance detected: {imbalance_ratio:.2f} of answers are the same")
    else:
        print(f"✅ Data balance looks good: {imbalance_ratio:.2f}")
    
    # WUPS Preview Test
    print(f"\n{'='*40}")
    print(f"WUPS METRICS PREVIEW TEST")
    print(f"{'='*40}")
    
    if sample_questions_for_wups:
        print(f"Testing WUPS computation with sample questions...")
        
        for i, sample in enumerate(sample_questions_for_wups, 1):
            print(f"\nSample {i}:")
            print(f"  Question: {sample['question']}")
            print(f"  All correct answers: {sample['answers']}")
            print(f"  Normalized answers: {sample['normalized_answers']}")
            
            # Test with first answer as prediction
            test_prediction = sample['normalized_answers'][0]
            test_references = [sample['normalized_answers']]
            
            try:
                # Test WUPS computation
                wups_00 = compute_wups([test_prediction], test_references, threshold=0.0)
                wups_09 = compute_wups([test_prediction], test_references, threshold=0.9)
                
                print(f"  Test prediction: '{test_prediction}'")
                print(f"  WUPS 0.0: {wups_00:.4f}")
                print(f"  WUPS 0.9: {wups_09:.4f}")
                
            except Exception as e:
                print(f"  ⚠️  WUPS calculation error: {e}")
                print(f"     Will use fallback WUPS implementation")
        
        print(f"\n✅ WUPS metrics preview completed")
    
    return answer_counts

def test_block_wise_masking(model, test_batch):
    """Test block-wise vision masking implementation"""
    print(f"\n{'='*60}")
    print(f"TESTING BLOCK-WISE VISION MASKING")
    print(f"{'='*60}")
    
    model.eval()
    
    # Move test batch to device
    device = next(model.parameters()).device
    test_batch_device = {}
    for key, value in test_batch.items():
        if isinstance(value, torch.Tensor):
            test_batch_device[key] = value.to(device)
        else:
            test_batch_device[key] = value
    
    with torch.no_grad():
        # Get original vision features
        original_features = model.vision_model(pixel_values=test_batch_device['pixel_values'][:1])
        original_vision = original_features.last_hidden_state
        
        print(f"Original vision features shape: {original_vision.shape}")
        print(f"Vision model info:")
        print(f"  Total patches: {original_vision.shape[1]}")
        print(f"  Expected: 197 (1 CLS + 196 patches from 14x14 grid)")
        
        # FIX: Check original feature statistics
        cls_token = original_vision[0, 0, :]
        patch_tokens = original_vision[0, 1:, :]
        
        print(f"  CLS token norm: {cls_token.norm().item():.6f}")
        print(f"  Patch tokens - min norm: {patch_tokens.norm(dim=1).min().item():.6f}")
        print(f"  Patch tokens - max norm: {patch_tokens.norm(dim=1).max().item():.6f}")
        print(f"  Patch tokens - mean norm: {patch_tokens.norm(dim=1).mean().item():.6f}")
        print(f"  Patch tokens - std: {patch_tokens.std().item():.6f}")
        
        # Apply block-wise masking
        from bartphobeit.model import apply_block_wise_vision_masking
        
        masked_vision, mask_indicators = apply_block_wise_vision_masking(
            original_vision,
            mask_ratio=model.vision_mask_ratio,
            block_size=model.vision_mask_block_size
        )
        
        print(f"Masked vision features shape: {masked_vision.shape}")
        print(f"Mask indicators shape: {mask_indicators.shape}")
        
        # Calculate masking statistics
        total_patches = mask_indicators.numel()
        masked_patches = mask_indicators.sum().item()
        masking_ratio = masked_patches / total_patches
        
        # More detailed analysis
        actual_patches = total_patches - 1  # Exclude CLS token
        actual_masked = mask_indicators[:, 1:].sum().item()  # Count only patch tokens
        actual_ratio = actual_masked / actual_patches
        
        print(f"Masking Statistics:")
        print(f"  Total tokens (including CLS): {total_patches}")
        print(f"  Actual image patches: {actual_patches}")
        print(f"  Masked patches: {actual_masked}")
        print(f"  Actual masking ratio: {actual_ratio:.3f}")
        print(f"  Target masking ratio: {model.vision_mask_ratio:.3f}")
        print(f"  Block size: {model.vision_mask_block_size}x{model.vision_mask_block_size}")
        
        # FIX: More thorough difference analysis
        # Check differences element-wise
        abs_diff = torch.abs(original_vision - masked_vision)
        total_diff = abs_diff.sum()
        max_diff = abs_diff.max()
        
        print(f"  Total absolute difference: {total_diff.item():.6f}")
        print(f"  Max absolute difference: {max_diff.item():.6f}")
        
        # Check specific masked patches
        masked_patch_indices = mask_indicators[0, 1:].nonzero().flatten()
        unmasked_patch_indices = (~mask_indicators[0, 1:]).nonzero().flatten()
        
        if len(masked_patch_indices) > 0:
            # Check norms of masked patches
            masked_norms_before = original_vision[0, masked_patch_indices + 1].norm(dim=1)
            masked_norms_after = masked_vision[0, masked_patch_indices + 1].norm(dim=1)
            
            print(f"  Masked patch analysis:")
            print(f"    Count: {len(masked_patch_indices)}")
            print(f"    Before masking - mean norm: {masked_norms_before.mean().item():.6f}")
            print(f"    After masking - mean norm: {masked_norms_after.mean().item():.6f}")
            print(f"    Max norm after masking: {masked_norms_after.max().item():.6f}")
            
            # Show some individual examples
            for i in range(min(3, len(masked_patch_indices))):
                idx = masked_patch_indices[i]
                before_norm = masked_norms_before[i].item()
                after_norm = masked_norms_after[i].item()
                print(f"      Patch {idx.item()}: {before_norm:.6f} -> {after_norm:.6f}")
        
        if len(unmasked_patch_indices) > 0:
            unmasked_norms_before = original_vision[0, unmasked_patch_indices + 1].norm(dim=1)
            unmasked_norms_after = masked_vision[0, unmasked_patch_indices + 1].norm(dim=1)
            
            print(f"  Unmasked patch analysis:")
            print(f"    Count: {len(unmasked_patch_indices)}")
            print(f"    Before masking - mean norm: {unmasked_norms_before.mean().item():.6f}")
            print(f"    After masking - mean norm: {unmasked_norms_after.mean().item():.6f}")
            print(f"    Should be identical: {torch.allclose(unmasked_norms_before, unmasked_norms_after)}")
        
        # FIX: Better success criteria
        if max_diff.item() > 1e-6:  # Use smaller threshold
            print(f"✅ Block-wise masking is working correctly")
        else:
            print(f"⚠️  No significant difference detected - masking may not be working")
            print(f"     Max difference: {max_diff.item():.2e} (threshold: 1e-6)")
        
        # Validate mask ratio with relaxed tolerance
        if actual_ratio >= model.vision_mask_ratio * 0.6:  # More relaxed tolerance
            print(f"✅ Masking ratio is acceptable ({actual_ratio:.3f} >= {model.vision_mask_ratio * 0.6:.3f})")
        else:
            print(f"⚠️  Masking ratio too low ({actual_ratio:.3f} < {model.vision_mask_ratio * 0.6:.3f})")
            print(f"     Consider adjusting block_size or using random masking fallback")
    
    return masked_vision, mask_indicators

def test_enhanced_model_forward(model, test_batch, device):
    """Test enhanced model forward pass with all features"""
    print(f"\n{'='*60}")
    print(f"TESTING ENHANCED MODEL FORWARD PASS")
    print(f"{'='*60}")
    
    # Move test batch to device
    for key, value in test_batch.items():
        if isinstance(value, torch.Tensor):
            test_batch[key] = value.to(device)
    
    model.train()  # Training mode for masking
    
    print(f"Testing training mode forward pass...")
    try:
        # Training forward pass
        train_outputs = model(
            pixel_values=test_batch['pixel_values'][:2],
            question_input_ids=test_batch['question_input_ids'][:2],
            question_attention_mask=test_batch['question_attention_mask'][:2],
            answer_input_ids=test_batch['answer_input_ids'][:2],
            answer_attention_mask=test_batch['answer_attention_mask'][:2]
        )
        
        print(f"✅ Training forward pass successful")
        print(f"  Loss: {train_outputs.loss.item():.4f}")
        print(f"  Logits shape: {train_outputs.logits.shape}")
        
        # Check for VQ-KD losses
        if hasattr(train_outputs, 'vq_losses') and train_outputs.vq_losses:
            print(f"  VQ-KD Losses:")
            for loss_name, loss_val in train_outputs.vq_losses.items():
                if torch.is_tensor(loss_val):
                    print(f"    {loss_name}: {loss_val.item():.4f}")
                else:
                    print(f"    {loss_name}: {loss_val:.4f}")
        
        # Check token accuracy if available
        if hasattr(train_outputs, 'token_accuracy'):
            print(f"  Token accuracy: {train_outputs.token_accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Training forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test inference mode
    model.eval()
    print(f"\nTesting inference mode...")
    
    try:
        with torch.no_grad():
            generated_ids = model(
                pixel_values=test_batch['pixel_values'][:1],
                question_input_ids=test_batch['question_input_ids'][:1],
                question_attention_mask=test_batch['question_attention_mask'][:1]
            )
            
            # Decode prediction
            pred_text = model.decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            print(f"✅ Inference successful")
            print(f"  Generated IDs shape: {generated_ids.shape}")
            print(f"  Sample prediction: '{pred_text}'")
            
            if 'answer_text' in test_batch:
                print(f"  Sample ground truth: '{test_batch['answer_text'][0]}'")
            
            # Test with multiple samples
            if test_batch['pixel_values'].size(0) > 1:
                multi_generated = model(
                    pixel_values=test_batch['pixel_values'][:3],
                    question_input_ids=test_batch['question_input_ids'][:3],
                    question_attention_mask=test_batch['question_attention_mask'][:3]
                )
                print(f"  Multi-sample generation shape: {multi_generated.shape}")
                
                # Show multiple predictions
                for i in range(min(3, multi_generated.size(0))):
                    pred = model.decoder_tokenizer.decode(multi_generated[i], skip_special_tokens=True)
                    print(f"    Prediction {i+1}: '{pred}'")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Enhanced main function with Full BART and BARTPhoBEiT features"""
    
    print(f"{'='*80}")
    print(f"ENHANCED VIETNAMESE VQA WITH FULL BART & BLOCK-WISE MASKING")
    print(f"{'='*80}")
    print(f"Features:")
    print(f"  🚀 Full BART encoder-decoder architecture")
    print(f"  🎯 Block-wise vision masking (40% patches)")
    print(f"  📊 WUPS 0.0 and WUPS 0.9 metrics")
    print(f"  🔧 VQ-KD Visual Tokenizer")
    print(f"  🌐 Multiway Transformer fusion")
    print(f"  🇻🇳 Vietnamese VQA optimization")
    
    # Load and validate configuration
    config = get_improved_config()
    config = validate_full_bart_config(config)
    
    print(f"\nUsing device: {config['device']}")
    
    # Test tokenizers
    question_tokenizer, answer_tokenizer = test_full_bart_tokenizers(config)
    
    # Load feature extractor
    print(f"\nLoading ViT feature extractor...")
    try:
        feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
        print(f"✅ Feature extractor loaded: {config['vision_model']}")
    except Exception as e:
        print(f"❌ Failed to load feature extractor: {e}")
        return
    
    # Load and prepare data
    print(f"\nLoading Vietnamese VQA dataset...")
    try:
        df = pd.read_csv('/home/tgng/coding/BARTphoBEIT_imple/text/evaluate_60k_data_balanced.csv')
        print(f"✅ Dataset loaded: {len(df)} samples")
        
        questions = prepare_data_from_dataframe(df)
        print(f"✅ Data prepared: {len(questions)} questions")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print(f"   Please check the dataset path and format")
        return
    
    # Enhanced data analysis with WUPS preview
    analyze_data_balance_with_wups_preview(questions)
    
    # Split data
    split_idx = int(0.8 * len(questions))
    train_questions = questions[:split_idx]
    val_questions = questions[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train questions: {len(train_questions):,}")
    print(f"  Validation questions: {len(val_questions):,}")
    
    # Test multiple answer normalization
    print(f"\nTesting multiple answer support...")
    if train_questions and 'all_correct_answers' in train_questions[0]:
        sample_answers = train_questions[0]['all_correct_answers']
        print(f"Sample question: {train_questions[0]['question']}")
        print(f"All 5 correct answers:")
        for i, ans in enumerate(sample_answers, 1):
            normalized = normalize_vietnamese_answer(ans)
            print(f"  {i}. '{ans}' → '{normalized}'")
    
    # Create enhanced datasets
    print(f"\nCreating enhanced datasets with Full BART support...")
    train_dataset = EnhancedVietnameseVQADataset(
        train_questions, config['image_dir'], question_tokenizer, 
        answer_tokenizer, feature_extractor, config['max_length'],
        use_augmentation=config.get('use_data_augmentation', True),
        augment_ratio=config.get('augment_ratio', 0.2),
        use_multiple_answers=True  # Enable multiple answers
    )
    train_dataset.set_training(True)
    
    val_dataset = EnhancedVietnameseVQADataset(
        val_questions, config['image_dir'], question_tokenizer, 
        answer_tokenizer, feature_extractor, config['max_length'],
        use_augmentation=False,
        use_multiple_answers=True  # Enable multiple answers
    )
    val_dataset.set_training(False)
    
    print(f"✅ Enhanced datasets created")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    print(f"\nCreating optimized data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,  # Reduced for stability
        pin_memory=True if config['device'] == 'cuda' else False,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if config['device'] == 'cuda' else False,
        drop_last=False
    )
    
    print(f"✅ Data loaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Initialize enhanced model with Full BART
    print(f"\nInitializing Enhanced BARTPhoBEiT model...")
    try:
        model = ImprovedVietnameseVQAModel(config)
        model = model.to(config['device'])
        
        print(f"✅ Enhanced model initialized")
        
        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nEnhanced Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model components
    print(f"\nTesting model components...")
    
    # Get test batch
    try:
        test_batch = next(iter(train_loader))
        print(f"✅ Test batch loaded: {test_batch['pixel_values'].shape[0]} samples")
        
        # Test block-wise masking
        test_block_wise_masking(model, test_batch)
        
        # Test enhanced model forward pass
        forward_success = test_enhanced_model_forward(model, test_batch, config['device'])
        
        if not forward_success:
            print(f"❌ Model testing failed, aborting training")
            return
            
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize enhanced trainer
    print(f"\nInitializing Enhanced Trainer with WUPS metrics...")
    try:
        trainer = ImprovedVQATrainer(model, train_loader, val_loader, config['device'], config)
        print(f"✅ Enhanced trainer initialized")
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # # Pre-training validation
    # print(f"\nRunning pre-training validation with WUPS metrics...")
    # try:
    #     pre_metrics, pre_predictions, pre_references = trainer.evaluate_with_wups()
    #     print(f"Pre-training Results:")
    #     print(f"  VQA Score: {pre_metrics.get('vqa_score', 0.0):.4f}")
    #     print(f"  WUPS 0.0: {pre_metrics.get('wups_0.0', 0.0):.4f}")
    #     print(f"  WUPS 0.9: {pre_metrics.get('wups_0.9', 0.0):.4f}")
    #     print(f"  Multi Fuzzy Accuracy: {pre_metrics.get('multi_fuzzy_accuracy', 0.0):.4f}")
    #     print(f"  Multi Exact Accuracy: {pre_metrics.get('multi_exact_accuracy', 0.0):.4f}")
        
    # except Exception as e:
    #     print(f"⚠️  Pre-training validation failed: {e}")
    #     print(f"   Proceeding with training anyway...")
    
    # Start enhanced training
    print(f"\n{'='*80}")
    print(f"STARTING ENHANCED BARTPHOBIT TRAINING")
    print(f"{'='*80}")
    print(f"Configuration Summary:")
    print(f"  🎯 Architecture: Full BART ({config['text_model']})")
    print(f"  🔧 Vision Masking: Block-wise {config.get('vision_mask_ratio', 0.4)*100:.0f}% (block size: {config.get('vision_mask_block_size', 4)})")
    print(f"  📊 WUPS Metrics: Enabled (0.0 and 0.9 thresholds)")
    print(f"  🧠 VQ-KD Tokenizer: {config.get('use_vqkd', False)}")
    print(f"  🌐 Multiway Layers: {config.get('num_multiway_layers', 6)}")
    print(f"  📚 Total epochs: {config['num_epochs']}")
    print(f"  📖 Stage 1 (Frozen): {config['stage1_epochs']} epochs")
    print(f"  📘 Stage 2 (Partial): {config['num_epochs'] - config['stage1_epochs']} epochs")
    print(f"  🎛️  Learning rates: Decoder={config['decoder_lr']:.2e}, Encoder={config['encoder_lr']:.2e}, Vision={config['vision_lr']:.2e}")
    print(f"  🎨 Data augmentation: {config.get('use_data_augmentation', False)}")
    print(f"  📊 Wandb logging: {config.get('use_wandb', False)}")
    
    try:
        # Start training
        best_score = trainer.train(config['num_epochs'])
        
        print(f"\n{'='*80}")
        print(f"ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"🏆 Final Results:")
        print(f"  Best VQA Score: {trainer.best_vqa_score:.4f}")
        print(f"  Best WUPS-0.9: {trainer.best_wups_09:.4f}")
        print(f"  Best Fuzzy Accuracy: {trainer.best_fuzzy_accuracy:.4f}")
        print(f"  Best Multi Exact Accuracy: {trainer.best_multi_exact_accuracy:.4f}")
        
        print(f"\n📁 Saved Models:")
        print(f"  🥇 best_vqa_model.pth - Best VQA Score model")
        print(f"  🥈 best_wups_model.pth - Best WUPS-0.9 model")
        print(f"  🥉 best_fuzzy_model.pth - Best Fuzzy Accuracy model")
        
        print(f"\n🎯 Enhanced Features Applied:")
        print(f"  ✅ Full BART encoder-decoder ({config['text_model']})")
        print(f"  ✅ Block-wise vision masking ({config.get('vision_mask_ratio', 0.4)*100:.0f}% patches)")
        print(f"  ✅ WUPS 0.0 and 0.9 metrics evaluation")
        print(f"  ✅ VQ-KD Visual Tokenizer")
        print(f"  ✅ Multiway Transformer fusion")
        print(f"  ✅ Enhanced Vietnamese VQA evaluation")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        print(f"Saving current state...")
        try:
            trainer.save_enhanced_checkpoint(
                trainer.global_step // len(train_loader), 
                {}, 
                is_best_vqa=False, 
                is_best_wups=False, 
                is_best_fuzzy=False
            )
            print(f"✅ Current state saved successfully")
        except Exception as save_e:
            print(f"❌ Failed to save current state: {save_e}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save emergency checkpoint
        try:
            print(f"Attempting to save emergency checkpoint...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'error': str(e)
            }, 'emergency_checkpoint.pth')
            print(f"✅ Emergency checkpoint saved")
        except:
            print(f"❌ Failed to save emergency checkpoint")
    
    print(f"\n{'='*80}")
    print(f"PROGRAM FINISHED")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()