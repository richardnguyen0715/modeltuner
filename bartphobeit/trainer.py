import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import json
import glob
import warnings  # ADD: Missing import
import math      # ADD: Missing import
from collections import defaultdict
import wandb
from datetime import datetime
from bartphobeit.model import ImprovedVietnameseVQAModel, normalize_vietnamese_answer, compute_metrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Install required packages for evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer  # type: ignore
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    BLEU_ROUGE_AVAILABLE = True
    print("✓ NLTK and rouge-score available for advanced metrics")
except ImportError:
    print("Warning: NLTK/Rouge not available. Install with: pip install nltk rouge-score")
    BLEU_ROUGE_AVAILABLE = False

class ImprovedVQATrainer:
    """Enhanced trainer with resume training functionality"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Training state
        self.current_stage = 1
        self.global_step = 0
        self.start_epoch = 0
        
        # For evaluation tracking
        self.best_vqa_score = 0
        self.best_wups_09 = 0
        self.best_fuzzy_accuracy = 0
        self.best_multi_exact_accuracy = 0
        
        # Checkpoint management
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup optimizers and schedulers
        self.setup_optimizers_and_schedulers()
        
        # ✨ NEW: Handle resume training
        if config.get('resume_training', False):
            self.resume_from_checkpoint()
        
        # Setup logging (after potential resume to maintain wandb consistency)
        self.setup_logging()
        
        # Evaluation metrics setup
        if BLEU_ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            self.smoothing_function = SmoothingFunction().method4
        
        # Enhanced logging
        print(f"Enhanced VQA Trainer initialized:")
        print(f"  Full BART model: {config['text_model']}")
        print(f"  VQ-KD Visual Tokenizer: {config.get('use_vqkd', False)}")
        print(f"  Block-wise vision masking: {config.get('use_unified_masking', False)}")
        print(f"  WUPS metrics: {config.get('calculate_wups', False)}")
        print(f"  Multiway Transformer layers: {config.get('num_multiway_layers', 6)}")
        print(f"  Resume training: {config.get('resume_training', False)}")
        if self.start_epoch > 0:
            print(f"  Resuming from epoch: {self.start_epoch + 1}")
            print(f"  Current stage: {self.current_stage}")
            print(f"  Global step: {self.global_step}")
    
    
    def setup_logging(self):
        """Setup enhanced wandb logging với resume support"""
        if self.config.get('use_wandb', False):
            try:
                # Generate unique run name
                run_name = f"BARTPhoBEiT_WUPS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if self.start_epoch > 0:
                    run_name += f"_resumed_epoch_{self.start_epoch + 1}"
                
                wandb.init(
                    project=self.config.get('project_name', 'BARTPhoBEiT-Vietnamese-VQA'),
                    config=self.config,
                    name=run_name,
                    tags=["BARTPhoBEiT", "Vietnamese-VQA", "WUPS", "Block-Masking", "Full-BART"],
                    resume="allow"  # Allow resuming wandb runs
                )
                
                # Log resume information
                if self.start_epoch > 0:
                    wandb.log({
                        'resumed_from_epoch': self.start_epoch + 1,
                        'resumed_global_step': self.global_step,
                        'resumed_stage': self.current_stage,
                        'resumed_best_vqa': self.best_vqa_score,
                        'resumed_best_wups': self.best_wups_09
                    })
                
                self.use_wandb = True
                print("✓ Enhanced wandb logging initialized")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def setup_optimizers_and_schedulers(self):
        """Enhanced optimizer setup với resume support"""
        
        # Group parameters by component with detailed analysis
        decoder_params = []
        encoder_params = []
        vision_params = []
        fusion_params = []
        vqkd_params = []
        
        param_analysis = defaultdict(int)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            param_count = param.numel()
            
            if 'text_decoder' in name or 'decoder' in name:
                decoder_params.append(param)
                param_analysis['decoder'] += param_count
            elif 'text_encoder' in name or 'encoder' in name and 'vision' not in name:
                encoder_params.append(param)
                param_analysis['text_encoder'] += param_count
            elif 'vision_model' in name:
                vision_params.append(param)
                param_analysis['vision'] += param_count
            elif 'visual_tokenizer' in name or 'vqkd' in name:
                vqkd_params.append(param)
                param_analysis['vqkd'] += param_count
            else:  # fusion, projection, and other components
                fusion_params.append(param)
                param_analysis['fusion'] += param_count
        
        # Setup parameter groups with optimized learning rates
        param_groups = []
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params, 
                'lr': self.config['decoder_lr'],
                'name': 'bart_decoder'
            })
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params, 
                'lr': self.config['encoder_lr'],
                'name': 'bart_encoder'
            })
        
        if vision_params:
            param_groups.append({
                'params': vision_params, 
                'lr': self.config['vision_lr'],
                'name': 'vision_encoder'
            })
        
        if vqkd_params:
            param_groups.append({
                'params': vqkd_params, 
                'lr': self.config['decoder_lr'] * 0.5,  # Lower LR for VQ-KD
                'name': 'vqkd_tokenizer'
            })
        
        if fusion_params:
            param_groups.append({
                'params': fusion_params, 
                'lr': self.config['decoder_lr'],
                'name': 'multiway_fusion'
            })
        
        # Create enhanced optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay'],
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Enhanced scheduler with warmup
        remaining_epochs = self.config['num_epochs'] - self.start_epoch
        remaining_steps = len(self.train_loader) * remaining_epochs
        total_steps = len(self.train_loader) * self.config['num_epochs']
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        # Adjust warmup if resuming
        if self.start_epoch > 0:
            completed_steps = len(self.train_loader) * self.start_epoch
            remaining_warmup = max(0, warmup_steps - completed_steps)
        else:
            remaining_warmup = warmup_steps
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=remaining_warmup,
            num_training_steps=remaining_steps
        )
        
        print(f"\nEnhanced Full BART optimizer setup:")
        print(f"  Total parameter groups: {len(param_groups)}")
        print(f"  Remaining training steps: {remaining_steps:,}")
        print(f"  Remaining warmup steps: {remaining_warmup:,}")
        
        for group in param_groups:
            count = param_analysis[group['name'].split('_')[0] if '_' in group['name'] else group['name']]
            print(f"  {group['name']}: {len(group['params'])} param tensors, {count:,} params, lr={group['lr']:.2e}")
    
    
    def evaluate_with_wups(self):
        """Enhanced evaluation with WUPS and comprehensive metrics"""
        self.model.eval()
        predictions = []
        all_correct_answers = []
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Enhanced Evaluation with WUPS", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Calculate validation loss if possible
                try:
                    loss_outputs = self.model(
                        pixel_values=batch['pixel_values'],
                        question_input_ids=batch['question_input_ids'],
                        question_attention_mask=batch['question_attention_mask'],
                        answer_input_ids=batch['answer_input_ids'],
                        answer_attention_mask=batch['answer_attention_mask']
                    )
                    total_loss += loss_outputs.loss.item()
                    num_batches += 1
                except Exception as e:
                    pass  # Skip loss calculation if it fails
                
                # Generate predictions for evaluation
                generated_ids = self.model(
                    pixel_values=batch['pixel_values'],
                    question_input_ids=batch['question_input_ids'],
                    question_attention_mask=batch['question_attention_mask']
                )
                
                # Decode predictions
                pred_texts = self.model.decoder_tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                predictions.extend(pred_texts)
                
                # Collect all correct answers for VQA score and WUPS computation
                if 'all_correct_answers' in batch:
                    all_correct_answers.extend(batch['all_correct_answers'])
                else:
                    # Fallback to single answer
                    all_correct_answers.extend([[ans] for ans in batch['answer_text']])
                
                # Update progress bar
                if num_batches > 0:
                    progress_bar.set_postfix({
                        'Val Loss': f"{total_loss/num_batches:.4f}",
                        'Samples': len(predictions)
                    })
        
        # Calculate comprehensive metrics including WUPS
        print(f"\nCalculating comprehensive metrics for {len(predictions)} predictions...")
        metrics = compute_metrics(predictions, all_correct_answers, self.model.decoder_tokenizer)
        
        # Add validation loss
        if num_batches > 0:
            metrics['val_loss'] = total_loss / num_batches
        
        return metrics, predictions, all_correct_answers
    
    def print_enhanced_metrics(self, metrics, epoch=None):
        """Enhanced metrics printing with WUPS and detailed analysis"""
        prefix = f"Epoch {epoch} " if epoch is not None else ""
        
        print(f"\n{prefix}Enhanced Evaluation Results:")
        print(f"  Primary Metrics:")
        print(f"    VQA Score: {metrics.get('vqa_score', 0.0):.4f} ⭐")
        print(f"    WUPS 0.0: {metrics.get('wups_0.0', 0.0):.4f} ⭐")
        print(f"    WUPS 0.9: {metrics.get('wups_0.9', 0.0):.4f} ⭐")
        print(f"    Multi Fuzzy Accuracy: {metrics.get('multi_fuzzy_accuracy', 0.0):.4f}")
        print(f"    Multi Exact Accuracy: {metrics.get('multi_exact_accuracy', 0.0):.4f}")
        
        print(f"  Secondary Metrics:")
        print(f"    Multi Token F1: {metrics.get('multi_token_f1', 0.0):.4f}")
        print(f"    Multi BLEU: {metrics.get('multi_bleu', 0.0):.4f}")
        print(f"    Multi ROUGE-L: {metrics.get('multi_rouge_l', 0.0):.4f}")
        
        if 'val_loss' in metrics:
            print(f"  Training Metrics:")
            print(f"    Validation Loss: {metrics['val_loss']:.4f}")
        
        # VQA Score distribution analysis
        if 'vqa_perfect_count' in metrics and 'total_samples' in metrics:
            total = metrics['total_samples']
            perfect = metrics.get('vqa_perfect_count', 0)
            partial = metrics.get('vqa_partial_count', 0)
            zero = metrics.get('vqa_zero_count', 0)
            
            print(f"  VQA Score Distribution:")
            print(f"    Perfect (1.0): {perfect}/{total} ({perfect/total*100:.1f}%)")
            print(f"    Partial (0<x<1): {partial}/{total} ({partial/total*100:.1f}%)")
            print(f"    Zero (0.0): {zero}/{total} ({zero/total*100:.1f}%)")
        
        # Enhanced diagnostics
        print(f"  Model Diagnostics:")
        if 'multi_exact_matches' in metrics:
            print(f"    Exact matches: {metrics['multi_exact_matches']}/{metrics.get('total_samples', 0)}")
        
        # Compare single vs multi-answer performance - FIX: Handle division by zero
        if 'single_exact_accuracy' in metrics:
            single_acc = metrics.get('single_exact_accuracy', 0)
            multi_acc = metrics.get('multi_exact_accuracy', 0)
            improvement = multi_acc - single_acc
            
            # FIX: Safe division to avoid ZeroDivisionError
            if single_acc > 0:
                improvement_percent = (improvement / single_acc) * 100
                print(f"    Multi-answer improvement: +{improvement:.4f} ({improvement_percent:.1f}%)")
            else:
                print(f"    Multi-answer improvement: +{improvement:.4f} (baseline: 0.0%)")
                print(f"    Single accuracy is 0, so percentage improvement is undefined")

    
    def save_enhanced_checkpoint(self, epoch, metrics, is_best_vqa=False, is_best_wups=False, is_best_fuzzy=False):
        """Enhanced checkpoint saving với resume support"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'current_stage': self.current_stage,
            'best_scores': {
                'vqa_score': self.best_vqa_score,
                'wups_0.9': self.best_wups_09,
                'fuzzy_accuracy': self.best_fuzzy_accuracy,
                'multi_exact_accuracy': self.best_multi_exact_accuracy
            },
            # Additional resume metadata
            'resume_metadata': {
                'save_time': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'model_architecture': 'BARTPhoBEiT',
                'training_complete': False
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save specialized best models
        saved_models = []
        
        if is_best_vqa:
            vqa_path = 'best_vqa_model.pth'
            torch.save(checkpoint, vqa_path)
            saved_models.append(f"VQA ({self.best_vqa_score:.4f})")
        
        if is_best_wups:
            wups_path = 'best_wups_model.pth'
            torch.save(checkpoint, wups_path)
            saved_models.append(f"WUPS-0.9 ({self.best_wups_09:.4f})")
        
        if is_best_fuzzy:
            fuzzy_path = 'best_fuzzy_model.pth'
            torch.save(checkpoint, fuzzy_path)
            saved_models.append(f"Fuzzy ({self.best_fuzzy_accuracy:.4f})")
        
        if saved_models:
            print(f"✓ New best model(s) saved: {', '.join(saved_models)}")
        
        # Keep only last N checkpoints
        self.cleanup_checkpoints()
        
        return checkpoint_path
    
    def find_latest_checkpoint(self):
        """Tìm checkpoint mới nhất"""
        checkpoint_patterns = [
            os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth'),
            'checkpoint_epoch_*.pth',
            'best_vqa_model.pth',
            'best_wups_model.pth',
            'best_fuzzy_model.pth'
        ]
        
        latest_checkpoint = None
        latest_epoch = -1
        
        for pattern in checkpoint_patterns:
            checkpoints = glob.glob(pattern)
            
            for checkpoint_path in checkpoints:
                try:
                    # Extract epoch from filename
                    filename = os.path.basename(checkpoint_path)
                    
                    if 'checkpoint_epoch_' in filename:
                        epoch_str = filename.replace('checkpoint_epoch_', '').replace('.pth', '')
                        epoch = int(epoch_str) - 1  # Convert to 0-based
                    elif 'best_' in filename:
                        # Try to load and get epoch from checkpoint
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        epoch = checkpoint.get('epoch', -1)
                    else:
                        continue
                    
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_checkpoint = checkpoint_path
                        
                except (ValueError, Exception) as e:
                    print(f"Warning: Could not parse checkpoint {checkpoint_path}: {e}")
                    continue
        
        return latest_checkpoint, latest_epoch

    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint với error handling tốt"""
        print(f"\n{'='*60}")
        print(f"LOADING CHECKPOINT: {checkpoint_path}")
        print(f"{'='*60}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Print checkpoint info
            print(f"Checkpoint information:")
            print(f"  File: {os.path.basename(checkpoint_path)}")
            print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"  Global step: {checkpoint.get('global_step', 'Unknown')}")
            print(f"  Stage: {checkpoint.get('current_stage', 'Unknown')}")
            
            # Load model state
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], 
                strict=self.config.get('resume_strict', True)
            )
            
            if missing_keys:
                print(f"Warning: Missing keys in model state dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in model state dict: {unexpected_keys}")
            
            print(f"✓ Model state loaded successfully")
            
            # Load training state
            self.start_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.current_stage = checkpoint.get('current_stage', 1)
            
            # Load best scores if available
            if self.config.get('resume_best_scores', True) and 'best_scores' in checkpoint:
                best_scores = checkpoint['best_scores']
                self.best_vqa_score = best_scores.get('vqa_score', 0)
                self.best_wups_09 = best_scores.get('wups_0.9', 0)
                self.best_fuzzy_accuracy = best_scores.get('fuzzy_accuracy', 0)
                
                print(f"✓ Best scores restored:")
                print(f"    VQA Score: {self.best_vqa_score:.4f}")
                print(f"    WUPS-0.9: {self.best_wups_09:.4f}")
                print(f"    Fuzzy Accuracy: {self.best_fuzzy_accuracy:.4f}")
            
            # Load optimizer state (optional)
            if not self.config.get('reset_optimizer_on_resume', False):
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"✓ Optimizer state loaded")
                    except Exception as e:
                        print(f"Warning: Could not load optimizer state: {e}")
                        print(f"Continuing with fresh optimizer...")
            else:
                print(f"✓ Optimizer state reset (as requested)")
            
            # Load scheduler state (optional)
            if not self.config.get('reset_scheduler_on_resume', False):
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print(f"✓ Scheduler state loaded")
                    except Exception as e:
                        print(f"Warning: Could not load scheduler state: {e}")
                        print(f"Continuing with fresh scheduler...")
            else:
                print(f"✓ Scheduler state reset (as requested)")
            
            # Load metrics if available
            if 'metrics' in checkpoint:
                last_metrics = checkpoint['metrics']
                print(f"✓ Last evaluation metrics:")
                print(f"    VQA Score: {last_metrics.get('vqa_score', 'N/A')}")
                print(f"    WUPS-0.9: {last_metrics.get('wups_0.9', 'N/A')}")
                print(f"    Multi Fuzzy: {last_metrics.get('multi_fuzzy_accuracy', 'N/A')}")
            
            print(f"\n✅ CHECKPOINT LOADED SUCCESSFULLY")
            print(f"Will resume training from epoch {self.start_epoch + 1}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def resume_from_checkpoint(self):
        """Main resume functionality"""
        resume_config = self.config.get('resume_from_checkpoint')
        auto_resume = self.config.get('auto_resume', True)
        
        checkpoint_path = None
        
        # Determine checkpoint path
        if resume_config == 'latest' or (resume_config is None and auto_resume):
            # Auto-detect latest checkpoint
            print("🔍 Auto-detecting latest checkpoint...")
            checkpoint_path, latest_epoch = self.find_latest_checkpoint()
            
            if checkpoint_path:
                print(f"✓ Found latest checkpoint: {os.path.basename(checkpoint_path)} (epoch {latest_epoch + 1})")
            else:
                print("ℹ️  No checkpoint found for auto-resume")
                return
                
        elif resume_config and os.path.exists(resume_config):
            # Specific checkpoint path provided
            checkpoint_path = resume_config
            print(f"📁 Using specified checkpoint: {os.path.basename(checkpoint_path)}")
            
        elif resume_config:
            print(f"❌ Specified checkpoint not found: {resume_config}")
            return
        
        # Load the checkpoint
        if checkpoint_path:
            success = self.load_checkpoint(checkpoint_path)
            
            if not success:
                print(f"⚠️  Failed to load checkpoint, starting from scratch")
                self.start_epoch = 0
                self.global_step = 0
                self.current_stage = 1
            else:
                # Handle stage transitions if necessary
                if self.current_stage == 1 and self.start_epoch >= self.config['stage1_epochs']:
                    print(f"🔄 Checkpoint was in stage 1, but should be in stage 2")
                    print(f"Transitioning to stage 2...")
                    self.model.partial_unfreeze(self.config['unfreeze_last_n_layers'])
                    self.current_stage = 2
                    
                    # Recreate optimizer for stage 2
                    print(f"Recreating optimizer for stage 2...")
                    self.setup_optimizers_and_schedulers()
    
    
    def save_final_checkpoint(self, final_metrics):
        """Save final checkpoint khi training hoàn thành"""
        final_checkpoint = {
            'epoch': self.config['num_epochs'] - 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': final_metrics,
            'current_stage': self.current_stage,
            'best_scores': {
                'vqa_score': self.best_vqa_score,
                'wups_0.9': self.best_wups_09,
                'fuzzy_accuracy': self.best_fuzzy_accuracy,
                'multi_exact_accuracy': self.best_multi_exact_accuracy
            },
            'resume_metadata': {
                'save_time': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'model_architecture': 'BARTPhoBEiT',
                'training_complete': True,  # Mark as completed
                'total_epochs': self.config['num_epochs']
            }
        }
        
        final_path = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(final_checkpoint, final_path)
        
        print(f"💾 Final model saved: {final_path}")
        return final_path
    
    
    def cleanup_checkpoints(self):
        """Keep only the last N checkpoints"""
        keep_n = self.config.get('keep_last_n_checkpoints', 5)
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth'))
        
        if len(checkpoints) > keep_n:
            # Sort by creation time
            checkpoints.sort(key=os.path.getctime)
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_n]:
                try:
                    os.remove(checkpoint)
                    print(f"Removed old checkpoint: {os.path.basename(checkpoint)}")
                except Exception as e:  # FIX: Better error handling
                    print(f"Failed to remove checkpoint {checkpoint}: {e}")


    def save_enhanced_predictions(self, predictions, all_correct_answers, epoch, metrics):
        """Save predictions with enhanced analysis"""
        if not self.config.get('save_predictions', True):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'epoch': epoch,
            'model_config': {
                'text_model': self.config['text_model'],
                'vision_model': self.config['vision_model'],
                'use_vqkd': self.config.get('use_vqkd', False),
                'use_unified_masking': self.config.get('use_unified_masking', False),
                'num_multiway_layers': self.config.get('num_multiway_layers', 6)
            },
            'metrics': metrics,
            'enhanced_analysis': {
                'vqa_score': metrics.get('vqa_score', 0),
                'wups_0_0': metrics.get('wups_0.0', 0),
                'wups_0_9': metrics.get('wups_0.9', 0),
                'multi_fuzzy_accuracy': metrics.get('multi_fuzzy_accuracy', 0),
                'multi_exact_accuracy': metrics.get('multi_exact_accuracy', 0)
            },
            'sample_results': []
        }
        
        # Enhanced sample analysis
        for i, (pred, correct_answers) in enumerate(zip(predictions[:100], all_correct_answers[:100])):
            sample_result = {
                'index': i,
                'prediction': pred,
                'normalized_prediction': normalize_vietnamese_answer(pred),
                'all_correct_answers': correct_answers,
                'normalized_correct_answers': [normalize_vietnamese_answer(ans) for ans in correct_answers]
            }
            
            # Calculate individual scores for this sample
            from bartphobeit.model import compute_vqa_score_single
            sample_result['vqa_score'] = compute_vqa_score_single(pred, correct_answers)
            
            # Check exact match with any correct answer
            norm_pred = normalize_vietnamese_answer(pred)
            norm_correct = [normalize_vietnamese_answer(ans) for ans in correct_answers]
            sample_result['exact_match'] = norm_pred in norm_correct
            
            results['sample_results'].append(sample_result)
        
        # Save to JSON
        results_file = f'enhanced_predictions_epoch_{epoch+1}_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Enhanced predictions saved to: {results_file}")
        
        # Save summary statistics
        summary_file = f'evaluation_summary_{timestamp}.json'
        summary = {
            'epoch': epoch,
            'total_samples': len(predictions),
            'primary_metrics': {
                'vqa_score': metrics.get('vqa_score', 0),
                'wups_0_0': metrics.get('wups_0.0', 0),
                'wups_0_9': metrics.get('wups_0.9', 0),
                'multi_fuzzy_accuracy': metrics.get('multi_fuzzy_accuracy', 0),
                'multi_exact_accuracy': metrics.get('multi_exact_accuracy', 0)
            },
            'model_info': {
                'architecture': 'BARTPhoBEiT',
                'text_model': self.config['text_model'],
                'vision_model': self.config['vision_model'],
                'enhancements': {
                    'vq_kd_tokenizer': self.config.get('use_vqkd', False),
                    'block_wise_masking': self.config.get('use_unified_masking', False),
                    'multiway_transformers': True,
                    'wups_evaluation': True
                }
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def train_epoch_enhanced(self, epoch):
        """Enhanced training epoch with VQ-KD loss and advanced monitoring"""
        self.model.train()
        total_loss = 0
        total_generation_loss = 0
        total_vqkd_loss = 0
        num_batches = len(self.train_loader)
        
        # Stage management with enhanced transitions
        if epoch == self.config['stage1_epochs']:
            print("\n" + "="*80)
            print("STAGE TRANSITION: Switching to Stage 2 - Partial Encoder Unfreezing")
            print("="*80)
            print(f"Unfreezing last {self.config['unfreeze_last_n_layers']} layers of encoders...")
            
            self.model.partial_unfreeze(self.config['unfreeze_last_n_layers'])
            self.setup_optimizers_and_schedulers()  # Recreate optimizer for new trainable params
            self.current_stage = 2
            
            print("Stage 2 optimizer groups:")
            for group in self.optimizer.param_groups:
                trainable_params = sum(p.numel() for p in group['params'] if p.requires_grad)
                print(f"  {group['name']}: {trainable_params:,} trainable params, lr={group['lr']:.2e}")
        
        stage_desc = f"Stage {self.current_stage} ({'Frozen' if self.current_stage == 1 else 'Partial Unfreeze'})"
        progress_bar = tqdm(self.train_loader, desc=f"{stage_desc} - Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with enhanced outputs
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                question_input_ids=batch['question_input_ids'],
                question_attention_mask=batch['question_attention_mask'],
                answer_input_ids=batch['answer_input_ids'],
                answer_attention_mask=batch['answer_attention_mask']
            )
            
            loss = outputs.loss
            
            # Track loss components if available
            generation_loss = loss.item()
            vqkd_loss = 0.0
            
            if hasattr(outputs, 'vq_losses') and outputs.vq_losses:
                vqkd_loss = outputs.vq_losses.get('vq_kd_loss', 0.0)
                if isinstance(vqkd_loss, torch.Tensor):
                    vqkd_loss = vqkd_loss.item()
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            total_generation_loss += generation_loss
            total_vqkd_loss += vqkd_loss
            self.global_step += 1
            
            # Enhanced progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_info = {
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss/(batch_idx+1):.4f}",
                'LR': f"{current_lr:.2e}",
                'Stage': self.current_stage
            }
            
            if vqkd_loss > 0:
                progress_info['VQ-KD'] = f"{vqkd_loss:.4f}"
            
            progress_bar.set_postfix(progress_info)
            
            # Enhanced wandb logging
            if self.use_wandb and batch_idx % 100 == 0:
                log_dict = {
                    'train_loss': loss.item(),
                    'train_generation_loss': generation_loss,
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'stage': self.current_stage,
                    'global_step': self.global_step
                }
                
                if vqkd_loss > 0:
                    log_dict['train_vqkd_loss'] = vqkd_loss
                
                # Add token accuracy if available
                if hasattr(outputs, 'token_accuracy'):
                    log_dict['train_token_accuracy'] = outputs.token_accuracy
                
                wandb.log(log_dict)
            
            # Enhanced step-level evaluation for debugging
            eval_freq = self.config.get('evaluate_every_n_steps', 5000)
            if epoch < 2 and eval_freq > 0:  # More frequent in first 2 epochs
                eval_freq = min(eval_freq, 2000)
            
            if eval_freq > 0 and self.global_step % eval_freq == 0:
                print(f"\n--- Step {self.global_step} Evaluation ---")
                step_metrics, step_predictions, step_references = self.evaluate_with_wups()
                
                print(f"Step {self.global_step} Results:")
                print(f"  VQA Score: {step_metrics.get('vqa_score', 0):.4f}")
                print(f"  WUPS 0.9: {step_metrics.get('wups_0.9', 0):.4f}")
                print(f"  Multi Fuzzy: {step_metrics.get('multi_fuzzy_accuracy', 0):.4f}")
                
                # Log to wandb
                if self.use_wandb:
                    step_log = {
                        'step_vqa_score': step_metrics.get('vqa_score', 0),
                        'step_wups_0_9': step_metrics.get('wups_0.9', 0),
                        'step_multi_fuzzy': step_metrics.get('multi_fuzzy_accuracy', 0),
                        'global_step': self.global_step
                    }
                    wandb.log(step_log)
                
                self.model.train()  # Back to training mode
        
        # Return comprehensive loss statistics
        return {
            'total_loss': total_loss / num_batches,
            'generation_loss': total_generation_loss / num_batches,
            'vqkd_loss': total_vqkd_loss / num_batches if total_vqkd_loss > 0 else 0.0
        }
    
    def train(self, num_epochs):
        """Enhanced training loop với resume support"""
        print(f"\n{'='*80}")
        print(f"STARTING ENHANCED BARTPhoBEiT TRAINING")
        print(f"{'='*80}")
        
        # Enhanced training info with resume details
        print(f"Training configuration:")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Stage 1 (Frozen encoders): epochs 1-{self.config['stage1_epochs']}")
        print(f"  Stage 2 (Partial unfreeze): epochs {self.config['stage1_epochs']+1}-{num_epochs}")
        
        if self.start_epoch > 0:
            print(f"  🔄 RESUMING from epoch: {self.start_epoch + 1}")
            print(f"  🔄 Current stage: {self.current_stage}")
            print(f"  🔄 Global step: {self.global_step}")
            print(f"  🔄 Best VQA score: {self.best_vqa_score:.4f}")
            print(f"  🔄 Best WUPS-0.9: {self.best_wups_09:.4f}")
            remaining_epochs = num_epochs - self.start_epoch
            print(f"  🔄 Remaining epochs: {remaining_epochs}")
        else:
            print(f"  🆕 Starting from scratch")
        
        print(f"  Full BART model: {self.config['text_model']}")
        print(f"  VQ-KD Visual Tokenizer: {self.config.get('use_vqkd', False)}")
        print(f"  Block-wise vision masking: {self.config.get('use_unified_masking', False)}")
        print(f"  WUPS evaluation: {self.config.get('calculate_wups', True)}")
        print(f"  Multiway Transformer layers: {self.config.get('num_multiway_layers', 6)}")
        
        # Training history
        training_history = {
            'epochs': [],
            'vqa_scores': [],
            'wups_0_9_scores': [],
            'fuzzy_accuracies': [],
            'train_losses': [],
            'resume_info': {
                'started_from_epoch': self.start_epoch,
                'initial_best_vqa': self.best_vqa_score,
                'initial_best_wups': self.best_wups_09
            }
        }
        
        # FIX: Initialize epoch variable to avoid UnboundLocalError
        epoch = self.start_epoch
        
        try:
            # Start training loop from resume point
            for epoch in range(self.start_epoch, num_epochs):
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch + 1}/{num_epochs}")
                if self.start_epoch > 0 and epoch == self.start_epoch:
                    print(f"🔄 RESUMED TRAINING")
                print(f"{'='*60}")
                
                # Enhanced training epoch
                train_metrics = self.train_epoch_enhanced(epoch)
                
                # Enhanced evaluation with WUPS
                print(f"\nRunning enhanced evaluation...")
                val_metrics, predictions, all_correct_answers = self.evaluate_with_wups()
                
                # Print comprehensive results
                self.print_enhanced_metrics(val_metrics, epoch + 1)
                
                # Training loss information
                print(f"\nTraining Loss Breakdown:")
                print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
                print(f"  Generation Loss: {train_metrics['generation_loss']:.4f}")
                if train_metrics['vqkd_loss'] > 0:
                    print(f"  VQ-KD Loss: {train_metrics['vqkd_loss']:.4f}")
                
                # Track best models
                current_vqa = val_metrics.get('vqa_score', 0)
                current_wups_09 = val_metrics.get('wups_0.9', 0)
                current_fuzzy = val_metrics.get('multi_fuzzy_accuracy', 0)
                current_exact = val_metrics.get('multi_exact_accuracy', 0)
                
                is_best_vqa = current_vqa > self.best_vqa_score
                is_best_wups = current_wups_09 > self.best_wups_09
                is_best_fuzzy = current_fuzzy > self.best_fuzzy_accuracy
                
                if is_best_vqa:
                    self.best_vqa_score = current_vqa
                if is_best_wups:
                    self.best_wups_09 = current_wups_09
                if is_best_fuzzy:
                    self.best_fuzzy_accuracy = current_fuzzy
                    self.best_multi_exact_accuracy = current_exact
                
                # Enhanced wandb logging
                if self.use_wandb:
                    comprehensive_log = {
                        'epoch': epoch + 1,
                        'train_total_loss': train_metrics['total_loss'],
                        'train_generation_loss': train_metrics['generation_loss'],
                        'val_vqa_score': current_vqa,
                        'val_wups_0_0': val_metrics.get('wups_0.0', 0),
                        'val_wups_0_9': current_wups_09,
                        'val_multi_fuzzy_accuracy': current_fuzzy,
                        'val_multi_exact_accuracy': current_exact,
                        'val_multi_token_f1': val_metrics.get('multi_token_f1', 0),
                        'val_multi_bleu': val_metrics.get('multi_bleu', 0),
                        'val_multi_rouge_l': val_metrics.get('multi_rouge_l', 0),
                        'best_vqa_score': self.best_vqa_score,
                        'best_wups_0_9': self.best_wups_09,
                        'best_fuzzy_accuracy': self.best_fuzzy_accuracy,
                        'current_stage': self.current_stage
                    }
                    
                    if train_metrics['vqkd_loss'] > 0:
                        comprehensive_log['train_vqkd_loss'] = train_metrics['vqkd_loss']
                    
                    # VQA score distribution
                    if 'vqa_perfect_count' in val_metrics:
                        total_samples = val_metrics['total_samples']
                        comprehensive_log.update({
                            'vqa_perfect_ratio': val_metrics['vqa_perfect_count'] / total_samples,
                            'vqa_partial_ratio': val_metrics.get('vqa_partial_count', 0) / total_samples,
                            'vqa_zero_ratio': val_metrics.get('vqa_zero_count', 0) / total_samples
                        })
                    
                    wandb.log(comprehensive_log)
                
                # Save enhanced checkpoints
                if (epoch + 1) % self.config.get('save_every_n_epochs', 1) == 0:
                    checkpoint_path = self.save_enhanced_checkpoint(
                        epoch, val_metrics, is_best_vqa, is_best_wups, is_best_fuzzy
                    )
                    print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
                
                # Save enhanced predictions
                self.save_enhanced_predictions(predictions, all_correct_answers, epoch, val_metrics)
                
                # Update training history
                training_history['epochs'].append(epoch + 1)
                training_history['vqa_scores'].append(current_vqa)
                training_history['wups_0_9_scores'].append(current_wups_09)
                training_history['fuzzy_accuracies'].append(current_fuzzy)
                training_history['train_losses'].append(train_metrics['total_loss'])
                
                # Print improvement notifications
                improvements = []
                if is_best_vqa:
                    improvements.append(f"VQA Score: {self.best_vqa_score:.4f}")
                if is_best_wups:
                    improvements.append(f"WUPS-0.9: {self.best_wups_09:.4f}")
                if is_best_fuzzy:
                    improvements.append(f"Fuzzy Accuracy: {self.best_fuzzy_accuracy:.4f}")
                
                if improvements:
                    print(f"\n🎉 New best results: {', '.join(improvements)}")
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user at epoch {epoch + 1}")
            print(f"Saving interruption checkpoint...")
            
            # Save emergency checkpoint
            try:
                emergency_checkpoint = {
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'config': self.config,
                    'current_stage': self.current_stage,
                    'best_scores': {
                        'vqa_score': self.best_vqa_score,
                        'wups_0.9': self.best_wups_09,
                        'fuzzy_accuracy': self.best_fuzzy_accuracy
                    },
                    'resume_metadata': {
                        'save_time': datetime.now().isoformat(),
                        'interrupted_at_epoch': epoch + 1,
                        'training_complete': False,
                        'reason': 'keyboard_interrupt'
                    }
                }
                
                interruption_path = f"interrupted_checkpoint_epoch_{epoch + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                torch.save(emergency_checkpoint, interruption_path)
                print(f"✅ Interruption checkpoint saved: {interruption_path}")
                print(f"💡 You can resume training by setting:")
                print(f"   config['resume_training'] = True")
                print(f"   config['resume_from_checkpoint'] = '{interruption_path}'")
                
            except Exception as save_e:
                print(f"❌ Failed to save interruption checkpoint: {save_e}")
            
            return self.best_vqa_score
        
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to save emergency checkpoint
            try:
                print(f"Attempting to save emergency checkpoint...")
                emergency_path = f"emergency_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'error': str(e),
                    'epoch': epoch,
                    'global_step': self.global_step
                }, emergency_path)
                print(f"✅ Emergency checkpoint saved: {emergency_path}")
            except Exception as emergency_e:
                print(f"❌ Failed to save emergency checkpoint: {emergency_e}")
            
            return self.best_vqa_score
        
        # Training completed successfully
        print(f"\n{'='*80}")
        print(f"ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        # Final evaluation and saving
        try:
            final_metrics, final_predictions, final_references = self.evaluate_with_wups()
            self.save_final_checkpoint(final_metrics)
            
            print(f"🏆 Final Results:")
            print(f"  Best VQA Score: {self.best_vqa_score:.4f}")
            print(f"  Best WUPS-0.9: {self.best_wups_09:.4f}")
            print(f"  Best Fuzzy Accuracy: {self.best_fuzzy_accuracy:.4f}")
            print(f"  Best Multi Exact Accuracy: {self.best_multi_exact_accuracy:.4f}")
            
        except Exception as e:
            print(f"Warning: Final evaluation failed: {e}")
        
        print(f"\n📁 Saved Models:")
        print(f"  🥇 best_vqa_model.pth - Best VQA Score model")
        print(f"  🥈 best_wups_model.pth - Best WUPS-0.9 model")
        print(f"  🥉 best_fuzzy_model.pth - Best Fuzzy Accuracy model")
        
        print(f"\n🎯 Enhanced Features Applied:")
        print(f"  ✅ Full BART encoder-decoder ({self.config['text_model']})")
        print(f"  ✅ Block-wise vision masking ({self.config.get('vision_mask_ratio', 0.4)*100:.0f}% patches)")
        print(f"  ✅ WUPS 0.0 and 0.9 metrics evaluation")
        print(f"  ✅ VQ-KD Visual Tokenizer")
        print(f"  ✅ Multiway Transformer fusion")
        print(f"  ✅ Enhanced Vietnamese VQA evaluation")
        print(f"  ✅ Resume training capability")
        
        # Save training history
        history_file = f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)
        
        print(f"  📊 Training history saved: {history_file}")
        
        # Final wandb summary
        if self.use_wandb:
            wandb.log({
                'final_best_vqa_score': self.best_vqa_score,
                'final_best_wups_0_9': self.best_wups_09,
                'final_best_fuzzy_accuracy': self.best_fuzzy_accuracy,
                'final_best_multi_exact_accuracy': self.best_multi_exact_accuracy,
                'total_epochs_completed': num_epochs,
                'training_completed': True
            })
            wandb.finish()
        
        return self.best_vqa_score