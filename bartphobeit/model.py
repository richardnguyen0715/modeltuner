import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, 
    ViTFeatureExtractor, ViTModel,
    MBartForConditionalGeneration
)
from transformers.modeling_outputs import BaseModelOutput
import re
import unicodedata
import random
from difflib import SequenceMatcher

class ImprovedMultimodalFusionLayer(nn.Module):
    """Enhanced multimodal fusion with proper dimension alignment"""
    
    def __init__(self, vision_dim, text_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        
        # Ensure proper dimension alignment
        assert hidden_dim == text_dim, f"hidden_dim ({hidden_dim}) should match text_dim ({text_dim})"
        
        # Dimension alignment - Critical for proper fusion
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),  # 768 -> 1024
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),    # 1024 -> 1024 (identity but with norm)
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-attention layers
        self.vision_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout_rate, batch_first=True
        )
        self.text_to_vision_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout_rate, batch_first=True
        )
        
        # Self-attention for final fusion
        self.fusion_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout_rate, batch_first=True
        )
        
        # Layer normalization and dropout
        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        print(f"Fusion layer: vision {vision_dim} -> {hidden_dim}, text {text_dim} -> {hidden_dim}")
        
    def forward(self, vision_features, text_features, text_attention_mask=None):
        # Project to same dimension with proper alignment
        vision_proj = self.vision_proj(vision_features)  # [batch, patches, hidden_dim]
        text_proj = self.text_proj(text_features)        # [batch, seq_len, hidden_dim]
        
        # Cross-attention: vision attends to text
        vision_attended, _ = self.vision_to_text_attn(
            vision_proj, text_proj, text_proj,
            key_padding_mask=~text_attention_mask.bool() if text_attention_mask is not None else None
        )
        vision_attended = self.vision_norm(vision_attended + vision_proj)
        vision_attended = self.dropout(vision_attended)
        
        # Cross-attention: text attends to vision  
        text_attended, _ = self.text_to_vision_attn(
            text_proj, vision_proj, vision_proj
        )
        text_attended = self.text_norm(text_attended + text_proj)
        text_attended = self.dropout(text_attended)
        
        # Concatenate attended features
        combined_features = torch.cat([vision_attended, text_attended], dim=1)
        
        # Create attention mask for combined features
        batch_size = vision_features.size(0)
        vision_patches = vision_features.size(1)
        
        # Vision attention mask (all ones since no padding)
        vision_mask = torch.ones(batch_size, vision_patches, device=vision_features.device, dtype=torch.bool)
        
        # Combine masks
        if text_attention_mask is not None:
            combined_mask = torch.cat([vision_mask, text_attention_mask.bool()], dim=1)
        else:
            text_len = text_features.size(1)
            text_mask = torch.ones(batch_size, text_len, device=text_features.device, dtype=torch.bool)
            combined_mask = torch.cat([vision_mask, text_mask], dim=1)
        
        # Self-attention for final fusion
        fused_features, _ = self.fusion_attn(
            combined_features, combined_features, combined_features,
            key_padding_mask=~combined_mask
        )
        fused_features = self.fusion_norm(fused_features + combined_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features, combined_mask

class ImprovedVietnameseVQAModel(nn.Module):
    """Enhanced Vietnamese VQA Model with proper BaseModelOutput handling"""
    
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        
        # Vision encoder (ViT)
        self.vision_model = ViTModel.from_pretrained(model_config['vision_model'])
        vision_dim = self.vision_model.config.hidden_size  # 768
        
        # Text encoder (PhoBERT)
        self.text_model = AutoModel.from_pretrained(model_config['text_model'])
        text_dim = self.text_model.config.hidden_size  # 1024
        
        # Text decoder (BARTPho)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(model_config['decoder_model'])
        self.text_decoder = MBartForConditionalGeneration.from_pretrained(model_config['decoder_model'])
        
        # Enhanced fusion layer with proper alignment
        hidden_dim = model_config.get('hidden_dim', 1024)
        dropout_rate = model_config.get('dropout_rate', 0.1)
        
        # Ensure dimensions are correct
        assert hidden_dim == text_dim, f"hidden_dim ({hidden_dim}) must match PhoBERT dim ({text_dim})"
        
        self.fusion_layer = ImprovedMultimodalFusionLayer(
            vision_dim, text_dim, hidden_dim, dropout_rate
        )
        
        # Output projection to normalize distribution for BART decoder
        decoder_dim = self.text_decoder.config.d_model  # Should be 1024 for BARTPho
        assert decoder_dim == hidden_dim, f"Decoder dim ({decoder_dim}) must match hidden dim ({hidden_dim})"
        
        # Enhanced projection layer for proper BART encoding
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(decoder_dim, decoder_dim),  # Additional transformation
            nn.LayerNorm(decoder_dim)
        )
        
        # Label smoothing loss
        self.label_smoothing = model_config.get('label_smoothing', 0.1)
        
        # Debugging flags
        self._debug_step = 0
        self._debug_freq = 100  # Debug every 100 steps
        
        # Initialize with frozen encoders
        self.freeze_encoders()
        
        # ✅ IMPROVED: Optimized pooling parameters for Vietnamese answers
        # Reduced target length from 32 to 16 for Vietnamese (most answers <10 tokens)
        self.pool_target_length = model_config.get('pool_target_length', 32)  # Reduced from 32
        
        # ✅ Attention pooling with learnable queries
        self.pool_queries = nn.Parameter(torch.randn(self.pool_target_length, decoder_dim))
        self.pool_attn = nn.MultiheadAttention(decoder_dim, num_heads=8, batch_first=True)
        
        # ✅ Learnable positional embeddings for pooled positions
        self.pool_pos_emb = nn.Embedding(self.pool_target_length, decoder_dim)
        
        # Initialize positional embeddings
        nn.init.normal_(self.pool_pos_emb.weight, std=0.02)
        nn.init.normal_(self.pool_queries, std=0.02)

        print(f"Enhanced Model components:")
        print(f"  Vision model: {model_config['vision_model']} (dim: {vision_dim})")
        print(f"  Text model: {model_config['text_model']} (dim: {text_dim})")
        print(f"  Decoder model: {model_config['decoder_model']} (dim: {decoder_dim})")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Label smoothing: {self.label_smoothing}")
        print(f"  Pool target length: {self.pool_target_length} (optimized for Vietnamese)")
    
    def freeze_encoders(self):
        """Freeze all encoder parameters"""
        frozen_params = 0
        
        for param in self.vision_model.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            
        for param in self.text_model.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            
        print(f"Encoders frozen ({frozen_params:,} parameters)")
    
    def partial_unfreeze(self, unfreeze_last_n_layers=2):
        """Enhanced partial unfreezing with better layer detection"""
        unfrozen_params = 0
        
        # Check PhoBERT structure with multiple possible layer names
        print(f"\nPhoBERT structure analysis:")
        phobert_layers = None
        
        if hasattr(self.text_model, 'encoder'):
            if hasattr(self.text_model.encoder, 'layer'):
                phobert_layers = self.text_model.encoder.layer
                print(f"  Found encoder.layer with {len(phobert_layers)} layers")
            elif hasattr(self.text_model.encoder, 'layers'):
                phobert_layers = self.text_model.encoder.layers
                print(f"  Found encoder.layers with {len(phobert_layers)} layers")
        
        if phobert_layers:
            # Unfreeze last N layers
            for i, layer in enumerate(phobert_layers[-unfreeze_last_n_layers:], len(phobert_layers)-unfreeze_last_n_layers):
                layer_params = sum(p.numel() for p in layer.parameters())
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                print(f"    Unfrozen PhoBERT layer {i}: {layer_params:,} params")
        else:
            print(f"  No encoder layers found in PhoBERT")
        
        # Check ViT structure with multiple possible layer names
        print(f"\nViT structure analysis:")
        vit_layers = None
        
        if hasattr(self.vision_model, 'encoder'):
            if hasattr(self.vision_model.encoder, 'layer'):
                vit_layers = self.vision_model.encoder.layer
                print(f"  Found encoder.layer with {len(vit_layers)} layers")
            elif hasattr(self.vision_model.encoder, 'layers'):
                vit_layers = self.vision_model.encoder.layers
                print(f"  Found encoder.layers with {len(vit_layers)} layers")
        
        if vit_layers:
            # Unfreeze last N layers
            for i, layer in enumerate(vit_layers[-unfreeze_last_n_layers:], len(vit_layers)-unfreeze_last_n_layers):
                layer_params = sum(p.numel() for p in layer.parameters())
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                print(f"    Unfrozen ViT layer {i}: {layer_params:,} params")
        else:
            print(f"  No encoder layers found in ViT")
        
        # Always unfreeze the pooler and layer norms
        pooler_params = 0
        for name, param in self.text_model.named_parameters():
            if any(keyword in name.lower() for keyword in ['pooler', 'layernorm', 'layer_norm']):
                if not param.requires_grad:  # Only count if wasn't already unfrozen
                    pooler_params += param.numel()
                param.requires_grad = True
        
        for name, param in self.vision_model.named_parameters():
            if any(keyword in name.lower() for keyword in ['pooler', 'layernorm', 'layer_norm']):
                if not param.requires_grad:  # Only count if wasn't already unfrozen
                    pooler_params += param.numel()
                param.requires_grad = True
        
        unfrozen_params += pooler_params
        
        print(f"\nPartial unfreezing summary:")
        print(f"  Total unfrozen parameters: {unfrozen_params:,}")
        print(f"  Pooler/LayerNorm parameters: {pooler_params:,}")
        print(f"  Last {unfreeze_last_n_layers} layers unfrozen successfully")
    
    def create_encoder_outputs(self, encoder_states):
        """Create proper encoder outputs for MBART decoder"""
        # MBART expects encoder outputs without explicit attention mask
        # The attention mask handling is done internally by the model
        return BaseModelOutput(
            last_hidden_state=encoder_states,
            hidden_states=None,
            attentions=None
        )
    
    def forward(self, pixel_values, question_input_ids, question_attention_mask, 
                answer_input_ids=None, answer_attention_mask=None):
        
        batch_size = pixel_values.size(0)
        self._debug_step += 1
        
        # Encode image
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [batch, patches, 768]
        
        # Encode question
        text_outputs = self.text_model(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, 1024]
        
        # Enhanced multimodal fusion
        fused_features, combined_attention_mask = self.fusion_layer(
            vision_features, text_features, question_attention_mask
        )  # [batch, combined_len, 1024], [batch, combined_len]
        
        # Project features for BART decoder
        encoder_states = self.output_proj(fused_features)  # [batch, combined_len, 1024]
        
        # Debug logging
        if self.training and self._debug_step % self._debug_freq == 0:
            print(f"Debug shapes (step {self._debug_step}):")
            print(f"  Vision features: {vision_features.shape}")
            print(f"  Text features: {text_features.shape}")
            print(f"  Fused features: {fused_features.shape}")
            print(f"  Encoder states: {encoder_states.shape}")
        
        if answer_input_ids is not None:  # Training mode
            # Validate token IDs
            vocab_size = self.text_decoder.config.vocab_size
            invalid_tokens = (answer_input_ids >= vocab_size) | (answer_input_ids < 0)
            if invalid_tokens.any():
                if self._debug_step % self._debug_freq == 0:
                    print(f"Warning: Found {invalid_tokens.sum()} invalid token IDs. Clamping to valid range.")
                answer_input_ids = torch.clamp(answer_input_ids, 0, vocab_size - 1)
            
            # ✅ OPTIMIZED: Use attention pooling with positional embeddings
            decoder_seq_len = min(answer_input_ids.size(1), self.pool_target_length)  # Cap at target length
            pooled_encoder_states = self.create_pooled_representation(
                encoder_states, combined_attention_mask, target_length=decoder_seq_len
            )  # [batch, decoder_seq_len, 1024]
            
            if self.training and self._debug_step % self._debug_freq == 0:
                print(f"  Pooled encoder states: {pooled_encoder_states.shape}")
                print(f"  Answer input ids: {answer_input_ids.shape}")
                print(f"  Target length optimized: {decoder_seq_len}")
            
            # Create proper encoder outputs
            encoder_outputs = self.create_encoder_outputs(pooled_encoder_states)
            
            if self.label_smoothing > 0:
                # Custom loss with label smoothing
                decoder_outputs = self.text_decoder(
                    input_ids=answer_input_ids[:, :decoder_seq_len],  # Truncate to match encoder
                    attention_mask=answer_attention_mask[:, :decoder_seq_len] if answer_attention_mask is not None else None,
                    encoder_outputs=encoder_outputs
                )
                
                logits = decoder_outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = answer_input_ids[:, 1:decoder_seq_len].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.decoder_tokenizer.pad_token_id,
                    label_smoothing=self.label_smoothing
                )
                
                # Calculate token accuracy
                with torch.no_grad():
                    predictions = torch.argmax(shift_logits, dim=-1)
                    mask = (shift_labels != self.decoder_tokenizer.pad_token_id)
                    correct_predictions = (predictions == shift_labels) & mask
                    token_accuracy = correct_predictions.sum().float() / mask.sum().float() if mask.sum() > 0 else 0.0
                
                if self.training and self._debug_step % self._debug_freq == 0:
                    print(f"  Loss: {loss.item():.4f}, Token accuracy: {token_accuracy:.4f}")
                
                class CustomDecoderOutput:
                    def __init__(self, loss, logits, token_accuracy=None):
                        self.loss = loss
                        self.logits = logits
                        self.token_accuracy = token_accuracy
                
                return CustomDecoderOutput(loss, logits, token_accuracy)
            
            else:
                # Standard loss
                decoder_outputs = self.text_decoder(
                    input_ids=answer_input_ids[:, :decoder_seq_len],
                    attention_mask=answer_attention_mask[:, :decoder_seq_len] if answer_attention_mask is not None else None,
                    encoder_outputs=encoder_outputs,
                    labels=answer_input_ids[:, :decoder_seq_len]
                )
                
                if self.training and self._debug_step % self._debug_freq == 0:
                    print(f"  Standard loss: {decoder_outputs.loss.item():.4f}")
                
                return decoder_outputs
            
        else:  # Inference mode
            # For generation, use optimized target length
            pooled_encoder_states = self.create_pooled_representation(
                encoder_states, combined_attention_mask, target_length=self.pool_target_length
            )  # [batch, pool_target_length, 1024]
            
            encoder_outputs = self.create_encoder_outputs(pooled_encoder_states)
            
            generated_ids = self.text_decoder.generate(
                encoder_outputs=encoder_outputs,
                max_length=self.pool_target_length,  # Use optimized length
                min_length=2,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
                eos_token_id=self.decoder_tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.2,
                length_penalty=1.0,
                no_repeat_ngram_size=2
            )
            return generated_ids
    
    def create_pooled_representation(self, encoder_states, attention_mask, target_length=None):
        """
        ✅ IMPROVED: Attention pooling with learnable queries and positional embeddings
        """
        batch_size, seq_len, hidden_dim = encoder_states.shape
        if target_length is None:
            target_length = self.pool_target_length

        # ✅ Method 1: Adaptive pooling for different target lengths
        if target_length != self.pool_target_length:
            # Use adaptive pooling for different lengths
            # Reshape for 1D adaptive pooling: [batch * hidden_dim, seq_len]
            reshaped = encoder_states.transpose(1, 2).contiguous()  # [batch, hidden_dim, seq_len]
            reshaped = reshaped.view(batch_size * hidden_dim, seq_len)  # [batch * hidden_dim, seq_len]
            
            # Apply adaptive average pooling
            pooled = F.adaptive_avg_pool1d(reshaped.unsqueeze(1), target_length).squeeze(1)
            # Reshape back: [batch, hidden_dim, target_length] -> [batch, target_length, hidden_dim]
            pooled = pooled.view(batch_size, hidden_dim, target_length).transpose(1, 2)
            
            return pooled
        
        # ✅ Method 2: Attention pooling with learnable queries (preferred)
        # Prepare queries: expand learnable queries to batch
        queries = self.pool_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, target_length, hidden_dim]
        
        # Add positional embeddings to queries
        pos_ids = torch.arange(target_length, device=queries.device)
        pos_emb = self.pool_pos_emb(pos_ids).unsqueeze(0)  # [1, target_length, hidden_dim]
        queries = queries + pos_emb  # Add positional information
        
        # Create key padding mask for attention
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        # Apply attention pooling: queries attend to encoder states
        pooled, attention_weights = self.pool_attn(
            queries,  # [batch, target_length, hidden_dim]
            encoder_states,  # [batch, seq_len, hidden_dim]  
            encoder_states,  # [batch, seq_len, hidden_dim]
            key_padding_mask=key_padding_mask  # [batch, seq_len]
        )
        
        # pooled: [batch, target_length, hidden_dim]
        return pooled

# ✅ ENHANCED: Comprehensive evaluation with BLEU/ROUGE logging
def compute_metrics(predictions, references, tokenizer=None):
    """Comprehensive evaluation metrics for Vietnamese VQA with enhanced logging"""
    
    # Normalize answers
    norm_preds = [normalize_vietnamese_answer(pred) for pred in predictions]
    norm_refs = [normalize_vietnamese_answer(ref) for ref in references]
    
    metrics = {}
    
    # 1. Exact match accuracy
    exact_matches = [pred == ref for pred, ref in zip(norm_preds, norm_refs)]
    metrics['exact_accuracy'] = sum(exact_matches) / len(exact_matches)
    
    # 2. Fuzzy accuracy with edit distance
    fuzzy_scores = []
    for pred, ref in zip(norm_preds, norm_refs):
        if pred == ref:
            fuzzy_scores.append(1.0)
        elif pred in ref or ref in pred:
            fuzzy_scores.append(0.8)  # Partial credit for substring match
        else:
            # Use sequence matcher for similarity
            similarity = SequenceMatcher(None, pred, ref).ratio()
            fuzzy_scores.append(similarity)
    
    metrics['fuzzy_accuracy'] = sum(fuzzy_scores) / len(fuzzy_scores)
    
    # 3. Token-level F1 score
    token_f1_scores = []
    token_precisions = []
    token_recalls = []
    
    for pred, ref in zip(norm_preds, norm_refs):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            token_f1_scores.append(1.0)
            token_precisions.append(1.0)
            token_recalls.append(1.0)
        elif len(pred_tokens) == 0:
            token_f1_scores.append(0.0)
            token_precisions.append(0.0)
            token_recalls.append(0.0)
        else:
            common_tokens = pred_tokens.intersection(ref_tokens)
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            token_precisions.append(precision)
            token_recalls.append(recall)
            token_f1_scores.append(f1)
    
    metrics['token_precision'] = sum(token_precisions) / len(token_precisions)
    metrics['token_recall'] = sum(token_recalls) / len(token_recalls)
    metrics['token_f1'] = sum(token_f1_scores) / len(token_f1_scores)
    
    # ✅ 4. BLEU score with enhanced logging
    bleu_scores = []
    bleu_available = False
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothing = SmoothingFunction().method4
        bleu_available = True
        
        for pred, ref in zip(norm_preds, norm_refs):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if len(ref_tokens) == 0:
                bleu_scores.append(0.0)
            else:
                try:
                    bleu = sentence_bleu([ref_tokens], pred_tokens, 
                                       smoothing_function=smoothing)
                    bleu_scores.append(bleu)
                except:
                    bleu_scores.append(0.0)
        
        metrics['bleu'] = sum(bleu_scores) / len(bleu_scores)
        
        # ✅ Enhanced BLEU analysis for debugging
        high_bleu_count = sum(1 for score in bleu_scores if score > 0.1)
        zero_bleu_count = sum(1 for score in bleu_scores if score == 0.0)
        
        metrics['bleu_high_count'] = high_bleu_count
        metrics['bleu_zero_count'] = zero_bleu_count
        metrics['bleu_nonzero_ratio'] = (len(bleu_scores) - zero_bleu_count) / len(bleu_scores)
        
    except ImportError:
        metrics['bleu'] = 0.0
        metrics['bleu_available'] = False
        print("⚠️  NLTK not available - BLEU score disabled")
    
    # ✅ 5. ROUGE-L score with enhanced logging
    rouge_scores = []
    rouge_available = False
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        rouge_available = True
        
        for pred, ref in zip(norm_preds, norm_refs):
            try:
                score = scorer.score(ref, pred)['rougeL'].fmeasure
                rouge_scores.append(score)
            except:
                rouge_scores.append(0.0)
        
        metrics['rouge_l'] = sum(rouge_scores) / len(rouge_scores)
        
        # ✅ Enhanced ROUGE analysis
        high_rouge_count = sum(1 for score in rouge_scores if score > 0.1)
        zero_rouge_count = sum(1 for score in rouge_scores if score == 0.0)
        
        metrics['rouge_high_count'] = high_rouge_count
        metrics['rouge_zero_count'] = zero_rouge_count
        metrics['rouge_nonzero_ratio'] = (len(rouge_scores) - zero_rouge_count) / len(rouge_scores)
        
    except ImportError:
        metrics['rouge_l'] = 0.0
        metrics['rouge_available'] = False
        print("⚠️  Rouge-score not available - ROUGE score disabled")
    
    # ✅ 6. Enhanced diagnostics for meaningless output detection
    empty_pred_count = sum(1 for pred in norm_preds if len(pred.strip()) == 0)
    repeated_pred_count = len(norm_preds) - len(set(norm_preds))  # How many predictions are duplicates
    
    metrics['empty_predictions'] = empty_pred_count
    metrics['repeated_predictions'] = repeated_pred_count
    metrics['unique_predictions'] = len(set(norm_preds))
    
    # Add counts for reference
    metrics['total_samples'] = len(predictions)
    metrics['exact_matches'] = sum(exact_matches)
    
    # ✅ Warning flags for debugging
    if bleu_available and metrics['bleu'] < 0.01:
        metrics['warning_low_bleu'] = True
    if rouge_available and metrics['rouge_l'] < 0.01:
        metrics['warning_low_rouge'] = True
    if empty_pred_count > len(predictions) * 0.1:
        metrics['warning_many_empty'] = True
    if repeated_pred_count > len(predictions) * 0.5:
        metrics['warning_repetitive'] = True
    
    return metrics

# Data augmentation functions
def augment_question(question, augment_ratio=0.2):
    """Simple Vietnamese question augmentation"""
    if random.random() > augment_ratio:
        return question
    
    words = question.split()
    if len(words) < 3:
        return question
    
    # Simple word shuffling (preserve first and last word)
    if len(words) > 4:
        middle = words[1:-1]
        random.shuffle(middle)
        return ' '.join([words[0]] + middle + [words[-1]])
    
    return question

def normalize_vietnamese_answer(answer):
    """Enhanced Vietnamese answer normalization"""
    if not isinstance(answer, str):
        answer = str(answer)
    
    # Unicode normalization
    answer = unicodedata.normalize('NFC', answer)
    
    # Convert to lowercase
    answer = answer.lower().strip()
    
    # Remove punctuation and extra whitespace
    answer = re.sub(r'[.,!?;:"\'()[\]{}]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Remove common Vietnamese articles and particles that don't affect meaning
    vietnamese_stopwords = ['các', 'của', 'và', 'là', 'trong', 'với', 'để', 'được', 'một', 'này', 'đó']
    words = answer.split()
    filtered_words = [w for w in words if w not in vietnamese_stopwords or len(words) <= 2]
    
    return ' '.join(filtered_words) if filtered_words else answer