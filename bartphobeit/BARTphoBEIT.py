import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import (
    AutoTokenizer, AutoModel, 
    ViTFeatureExtractor, ViTModel,
    BartForConditionalGeneration
)
from transformers.modeling_outputs import BaseModelOutput
from PIL import Image
import pandas as pd
import numpy as np
import ast
import os
from tqdm import tqdm
import json
from collections import defaultdict
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class VietnameseVQADataset(Dataset):
    """Dataset class for Vietnamese VQA"""
    
    def __init__(self, questions, image_dir, question_tokenizer, answer_tokenizer, feature_extractor, max_length=128, transform=None):
        self.questions = questions
        self.image_dir = image_dir
        self.question_tokenizer = question_tokenizer
        self.answer_tokenizer = answer_tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.transform = transform
        
        # Default image transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question_data = self.questions[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, question_data['image_name'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Create dummy image if file not found
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process image with ViT feature extractor
        image_inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        # Tokenize question with question tokenizer (PhoBERT)
        question = question_data['question']
        question_encoding = self.question_tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize answer with answer tokenizer (BART)
        answer = question_data['ground_truth']
        answer_encoding = self.answer_tokenizer(
            answer,
            max_length=32,  # Answers are typically shorter
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'question_input_ids': question_encoding['input_ids'].squeeze(0),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(0),
            'answer_input_ids': answer_encoding['input_ids'].squeeze(0),
            'answer_attention_mask': answer_encoding['attention_mask'].squeeze(0),
            'question_text': question,
            'answer_text': answer,
            'question_id': question_data.get('question_id', idx)
        }

class MultimodalFusionLayer(nn.Module):
    """Multimodal fusion layer for combining vision and language features"""
    
    def __init__(self, vision_dim, text_dim, hidden_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion_layer = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, vision_features, text_features):
        # Project features to same dimension
        vision_proj = self.vision_proj(vision_features)  # [batch, patches, hidden_dim]
        text_proj = self.text_proj(text_features)        # [batch, seq_len, hidden_dim]
        
        # Concatenate vision and text features
        combined_features = torch.cat([vision_proj, text_proj], dim=1)
        
        # Self-attention for fusion
        fused_features, _ = self.fusion_layer(
            combined_features, combined_features, combined_features
        )
        
        # Residual connection and normalization
        fused_features = self.norm(fused_features + combined_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features

class VietnameseVQAModel(nn.Module):
    """Vietnamese VQA Model inspired by BARTPhoBEiT architecture"""
    
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        
        # Vision encoder (ViT)
        self.vision_model = ViTModel.from_pretrained(model_config['vision_model'])
        vision_dim = self.vision_model.config.hidden_size
        
        # Text encoder (Vietnamese BERT/PhoBERT)
        self.text_model = AutoModel.from_pretrained(model_config['text_model'])
        text_dim = self.text_model.config.hidden_size
        
        # Text decoder (Vietnamese BART/BARTPho)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(model_config['decoder_model'])
        self.text_decoder = BartForConditionalGeneration.from_pretrained(model_config['decoder_model'])

        # Fusion layer
        hidden_dim = model_config.get('hidden_dim', 768)
        self.fusion_layer = MultimodalFusionLayer(vision_dim, text_dim, hidden_dim)
        
        # Output projection
        decoder_dim = self.text_decoder.config.d_model
        self.output_proj = nn.Linear(hidden_dim, decoder_dim)
        
        # Freeze vision and text encoders initially
        if model_config.get('freeze_encoders', True):
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        print(f"Model components:")
        print(f"  Vision model: {model_config['vision_model']} (dim: {vision_dim})")
        print(f"  Text model: {model_config['text_model']} (dim: {text_dim})")
        print(f"  Decoder model: {model_config['decoder_model']} (dim: {decoder_dim})")
        print(f"  Hidden dim: {hidden_dim}")
    
    def forward(self, pixel_values, question_input_ids, question_attention_mask, 
                answer_input_ids=None, answer_attention_mask=None):
        
        batch_size = pixel_values.size(0)
        
        # Encode image
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [batch, patches, vision_dim]
        
        # Encode question
        text_outputs = self.text_model(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, text_dim]
        
        # Multimodal fusion
        fused_features = self.fusion_layer(vision_features, text_features)
        
        # Project to decoder dimension
        decoder_inputs = self.output_proj(fused_features)
        
        if answer_input_ids is not None:  # Training mode
            # Use the answer sequence length to avoid dimension mismatch
            target_seq_len = answer_input_ids.size(1)
            
            # Average pool the fused features and expand to match target sequence length
            pooled = decoder_inputs.mean(dim=1, keepdim=True)  # [batch, 1, decoder_dim]
            encoder_hidden_states = pooled.expand(-1, target_seq_len, -1)  # [batch, target_seq_len, decoder_dim]
            
            # Validate token IDs to avoid out-of-vocabulary errors
            vocab_size = self.text_decoder.config.vocab_size
            answer_input_ids = torch.clamp(answer_input_ids, 0, vocab_size - 1)
            
            # Create encoder outputs with proper dimensions
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
            
            try:
                decoder_outputs = self.text_decoder(
                    input_ids=answer_input_ids,
                    attention_mask=answer_attention_mask,
                    encoder_outputs=encoder_outputs,
                    labels=answer_input_ids
                )
                return decoder_outputs
            except Exception as e:
                print(f"Decoder error: {e}")
                print(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
                print(f"Answer input ids shape: {answer_input_ids.shape}")
                print(f"Answer attention mask shape: {answer_attention_mask.shape}")
                raise e
                
        else:  # Inference mode
            # For generation, we need a different approach
            # Use a fixed sequence length for encoder outputs
            fixed_seq_len = 32  # Match typical answer length
            
            # Average pool and expand to fixed length
            pooled = decoder_inputs.mean(dim=1, keepdim=True)
            encoder_hidden_states = pooled.expand(-1, fixed_seq_len, -1)
            
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
            
            generated_ids = self.text_decoder.generate(
                encoder_outputs=encoder_outputs,
                max_length=32,
                min_length=1,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
                eos_token_id=self.decoder_tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            return generated_ids


class VQAEvaluator:
    """Evaluator for VQA models with Vietnamese-specific metrics"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def normalize_answer(self, answer):
        """Normalize Vietnamese answer text"""
        if not isinstance(answer, str):
            answer = str(answer)
        
        # Convert to lowercase and strip
        answer = answer.lower().strip()
        
        # Remove common Vietnamese stop words and punctuation
        answer = re.sub(r'[.,!?;]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
    
    def calculate_metrics(self, predictions, ground_truths):
        """Calculate evaluation metrics"""
        
        # Normalize answers
        norm_predictions = [self.normalize_answer(pred) for pred in predictions]
        norm_ground_truths = [self.normalize_answer(gt) for gt in ground_truths]
        
        # Exact match accuracy
        exact_matches = [pred == gt for pred, gt in zip(norm_predictions, norm_ground_truths)]
        accuracy = np.mean(exact_matches)
        
        # Token-level F1 score
        f1_scores = []
        precisions = []
        recalls = []
        
        for pred, gt in zip(norm_predictions, norm_ground_truths):
            pred_tokens = set(pred.split())
            gt_tokens = set(gt.split())
            
            if len(pred_tokens) == 0 and len(gt_tokens) == 0:
                f1_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            elif len(pred_tokens) == 0:
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            elif len(gt_tokens) == 0:
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(1.0)
            else:
                common_tokens = pred_tokens.intersection(gt_tokens)
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
        
        return {
            'accuracy': accuracy,
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores),
            'exact_match_count': sum(exact_matches),
            'total_count': len(predictions)
        }

class VQATrainer:
    """Trainer class for Vietnamese VQA model"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.get('learning_rate', 3e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get('scheduler_step', 10),
            gamma=config.get('scheduler_gamma', 0.1)
        )
        
        self.evaluator = VQAEvaluator(model.decoder_tokenizer)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                question_input_ids=batch['question_input_ids'],
                question_attention_mask=batch['question_attention_mask'],
                answer_input_ids=batch['answer_input_ids'],
                answer_attention_mask=batch['answer_attention_mask']
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss/(batch_idx+1):.4f}"
            })
        
        return total_loss / num_batches
    
    def evaluate(self):
        """Enhanced evaluation with VQA score"""
        # Import compute_metrics function
        from bartphobeit.model import compute_metrics
        
        self.model.eval()
        predictions = []
        all_correct_answers = []
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Calculate validation loss
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
                except:
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
                
                # Collect all correct answers for VQA score computation
                if 'all_correct_answers' in batch:
                    all_correct_answers.extend(batch['all_correct_answers'])
                else:
                    # Fallback to single answer
                    all_correct_answers.extend([[ans] for ans in batch['answer_text']])
                
                # Update progress bar
                if num_batches > 0:
                    progress_bar.set_postfix({'Val Loss': f"{total_loss/num_batches:.4f}"})
        
        # Calculate comprehensive metrics including VQA score
        metrics = compute_metrics(predictions, all_correct_answers, self.model.decoder_tokenizer)
        
        # Add validation loss to metrics
        if num_batches > 0:
            metrics['val_loss'] = total_loss / num_batches
        
        # Enhanced metrics reporting
        print(f"\nValidation Results:")
        print(f"  VQA Score: {metrics.get('vqa_score', 0.0):.4f}")
        print(f"  Multi Exact Accuracy: {metrics.get('multi_exact_accuracy', 0.0):.4f}")
        print(f"  Multi Fuzzy Accuracy: {metrics.get('multi_fuzzy_accuracy', 0.0):.4f}")
        print(f"  Multi Token F1: {metrics.get('multi_token_f1', 0.0):.4f}")
        print(f"  Multi BLEU: {metrics.get('multi_bleu', 0.0):.4f}")
        
        # VQA score distribution analysis
        if 'vqa_perfect_count' in metrics:
            total_samples = metrics['total_samples']
            print(f"  VQA Score Distribution:")
            print(f"    Perfect (1.0): {metrics['vqa_perfect_count']}/{total_samples} ({metrics.get('perfect_ratio', 0)*100:.1f}%)")
            print(f"    Partial (0<x<1): {metrics['vqa_partial_count']}/{total_samples} ({metrics.get('partial_ratio', 0)*100:.1f}%)")
            print(f"    Zero (0.0): {metrics['vqa_zero_count']}/{total_samples} ({metrics.get('zero_ratio', 0)*100:.1f}%)")
        
        if 'val_loss' in metrics:
            print(f"  Validation Loss: {metrics['val_loss']:.4f}")
        
        return metrics, predictions, all_correct_answers
    
    def train(self, num_epochs):
        """Enhanced training loop with VQA score tracking"""
        best_vqa_score = 0
        best_multi_fuzzy = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate with VQA score
            val_metrics, predictions, all_correct_answers = self.evaluate()
            
            # Learning rate scheduling
            if hasattr(self, 'scheduler'):
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        # Use VQA score for plateau scheduling
                        self.scheduler.step(val_metrics.get('vqa_score', 0))
                    else:
                        self.scheduler.step()
            
            # Print comprehensive results
            print(f"\nTraining Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  VQA Score: {val_metrics.get('vqa_score', 0.0):.4f}")
            print(f"  Multi Fuzzy Accuracy: {val_metrics.get('multi_fuzzy_accuracy', 0.0):.4f}")
            print(f"  Multi Exact Accuracy: {val_metrics.get('multi_exact_accuracy', 0.0):.4f}")
            print(f"  Multi Token F1: {val_metrics.get('multi_token_f1', 0.0):.4f}")
            
            # Save best model based on VQA score (primary metric)
            current_vqa_score = val_metrics.get('vqa_score', 0)
            current_fuzzy_score = val_metrics.get('multi_fuzzy_accuracy', 0)
            
            if current_vqa_score > best_vqa_score:
                best_vqa_score = current_vqa_score
                torch.save(self.model.state_dict(), 'best_vqa_model.pth')
                print(f"New best VQA score model saved: {best_vqa_score:.4f}")
                
                # Save detailed results for best model
                results = {
                    'epoch': epoch + 1,
                    'predictions': predictions[:100],  # Save first 100 for analysis
                    'metrics': val_metrics,
                    'model_path': 'best_vqa_model.pth'
                }
                
                with open('best_vqa_results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Also save best fuzzy accuracy model as backup
            if current_fuzzy_score > best_multi_fuzzy:
                best_multi_fuzzy = current_fuzzy_score
                torch.save(self.model.state_dict(), 'best_fuzzy_model.pth')
                print(f"New best fuzzy accuracy model saved: {best_multi_fuzzy:.4f}")
            
            # Wandb logging if available
            if hasattr(self, 'use_wandb') and self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_metrics.get('val_loss', 0),
                        'vqa_score': val_metrics.get('vqa_score', 0),
                        'multi_exact_accuracy': val_metrics.get('multi_exact_accuracy', 0),
                        'multi_fuzzy_accuracy': val_metrics.get('multi_fuzzy_accuracy', 0),
                        'multi_token_f1': val_metrics.get('multi_token_f1', 0),
                        'multi_bleu': val_metrics.get('multi_bleu', 0),
                        'vqa_perfect_ratio': val_metrics.get('perfect_ratio', 0),
                        'vqa_partial_ratio': val_metrics.get('partial_ratio', 0),
                        'vqa_zero_ratio': val_metrics.get('zero_ratio', 0),
                    })
                except:
                    pass
            
            # Save final epoch results
            if epoch == num_epochs - 1:
                final_results = {
                    'final_epoch': epoch + 1,
                    'predictions': predictions,
                    'all_correct_answers': all_correct_answers,
                    'metrics': val_metrics,
                    'best_vqa_score': best_vqa_score,
                    'best_multi_fuzzy': best_multi_fuzzy
                }
                
                with open('final_evaluation_results.json', 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"Best VQA Score achieved: {best_vqa_score:.4f}")
        print(f"Best Multi Fuzzy Accuracy: {best_multi_fuzzy:.4f}")
        print(f"Models saved:")
        print(f"  - best_vqa_model.pth (VQA Score: {best_vqa_score:.4f})")
        print(f"  - best_fuzzy_model.pth (Fuzzy Accuracy: {best_multi_fuzzy:.4f})")
        
        return best_vqa_score

def prepare_data_from_dataframe(df):
    """Convert your existing dataframe format to questions list with multiple correct answers"""
    print(f"Dataframe len: {len(df)}")
    
    # Parse answers if they're string representations of lists
    if isinstance(df['answers'].iloc[0], str):
        try:
            df['answers'] = df['answers'].apply(ast.literal_eval)
        except Exception:
            # fallback: wrap as single-element list
            df['answers'] = df['answers'].apply(lambda x: [x])
    
    # Create questions list with all 5 correct answers
    questions = []
    for idx, row in df.iterrows():
        answers_list = row['answers'] if isinstance(row['answers'], list) else [row['answers']]
        
        questions.append({
            'question_id': row.get('index', idx),
            'image_name': row['image_name'],
            'question': row['question'],
            'answers': answers_list,  # Keep all 5 answers
            'ground_truth': answers_list[0] if answers_list else "",  # Primary answer for training
            'all_correct_answers': answers_list  # All correct answers for evaluation
        })
    
    print(f"Loaded {len(questions)} questions from dataset")
    print(f"Average answers per question: {sum(len(q['answers']) for q in questions) / len(questions):.2f}")
    return questions