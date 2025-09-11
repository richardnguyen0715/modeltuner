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
import numpy as np
from difflib import SequenceMatcher
import math

class MultiWayTransformerLayer(nn.Module):
    """
    Multiway Transformer layer theo paper BARTPhoBEiT
    - Shared self-attention cho tất cả modalities
    - Separate experts (FFNs) cho vision, language, và vision-language fusion
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout_rate=0.1, is_fusion_layer=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_fusion_layer = is_fusion_layer
        
        # Shared self-attention module
        self.shared_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Vision expert FFN
        self.vision_expert = self._create_expert_ffn(hidden_dim, dropout_rate)
        
        # Language expert FFN
        self.language_expert = self._create_expert_ffn(hidden_dim, dropout_rate)
        
        # Vision-Language expert FFN (chỉ cho fusion layers - 3 layers cuối)
        if is_fusion_layer:
            self.vision_language_expert = self._create_expert_ffn(hidden_dim, dropout_rate)
            print(f"  Vision-Language expert added to fusion layer")
        
        self.expert_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        print(f"MultiWay layer: hidden_dim={hidden_dim}, fusion_layer={is_fusion_layer}")
        
    def _create_expert_ffn(self, hidden_dim, dropout_rate):
        """Create expert FFN theo BART/Transformer architecture"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x, modality_mask, attention_mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] - Combined vision + text tokens
            modality_mask: [batch_size, seq_len] - 0: vision, 1: language, 2: fusion
            attention_mask: [batch_size, seq_len] - Attention mask for padding
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. Shared self-attention
        if attention_mask is not None:
            # Convert attention mask: True for valid tokens, False for padding
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        attn_output, attn_weights = self.shared_attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        
        # Residual connection + Layer norm
        x = self.attention_norm(x + self.dropout(attn_output))
        
        # 2. Route tokens to appropriate experts
        expert_output = torch.zeros_like(x)
        
        # Process each sample in batch
        for batch_idx in range(batch_size):
            # Vision tokens -> vision expert
            vision_mask = (modality_mask[batch_idx] == 0)
            if vision_mask.any():
                vision_tokens = x[batch_idx, vision_mask]  # [num_vision_tokens, hidden_dim]
                if vision_tokens.numel() > 0:
                    expert_output[batch_idx, vision_mask] = self.vision_expert(vision_tokens)
            
            # Language tokens -> language expert
            language_mask = (modality_mask[batch_idx] == 1)
            if language_mask.any():
                language_tokens = x[batch_idx, language_mask]  # [num_lang_tokens, hidden_dim]
                if language_tokens.numel() > 0:
                    expert_output[batch_idx, language_mask] = self.language_expert(language_tokens)
            
            # Fusion tokens -> vision-language expert (nếu có)
            if self.is_fusion_layer:
                fusion_mask = (modality_mask[batch_idx] == 2)
                if fusion_mask.any():
                    fusion_tokens = x[batch_idx, fusion_mask]  # [num_fusion_tokens, hidden_dim]
                    if fusion_tokens.numel() > 0:
                        expert_output[batch_idx, fusion_mask] = self.vision_language_expert(fusion_tokens)
        
        # Residual connection + Layer norm
        output = self.expert_norm(x + expert_output)
        
        return output, attn_weights

class VQKDVisualTokenizer(nn.Module):
    """
    Vector Quantized Knowledge Distillation Visual Tokenizer
    Theo paper BARTPhoBEiT - BEIT-2 visual tokenizer
    """
    
    def __init__(self, vocab_size=8192, embed_dim=768, teacher_dim=768):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.teacher_dim = teacher_dim
        # Codebook V ∈ R^{K×D}
        self.codebook = nn.Embedding(vocab_size, embed_dim)
        
        # Teacher model (sử dụng pre-trained ViT)
        self.teacher_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.teacher_model.parameters():
            param.requires_grad = False  # Freeze teacher
        
        # Decoder để reconstruct teacher features
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, teacher_dim)
        
        # Initialize codebook
        nn.init.normal_(self.codebook.weight, std=0.02)
        
        print(f"VQ-KD Visual Tokenizer: vocab_size={vocab_size}, embed_dim={embed_dim}")
    
    def forward(self, images, return_loss=True):
        batch_size = images.size(0)
        
        # 1. Get teacher features (frozen)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(pixel_values=images)
            teacher_features = teacher_outputs.last_hidden_state  # [batch, patches, teacher_dim]
        
        # 2. Encode patches -> h_i (sử dụng teacher features làm encoder features)
        patch_features = teacher_features  # [batch, patches, teacher_dim]
        
        if patch_features.size(-1) != self.embed_dim:
            # Project teacher features to embed_dim
            patch_features = F.linear(patch_features, 
                                    torch.randn(self.embed_dim, teacher_features.size(-1), 
                                              device=patch_features.device) * 0.02)
        
        # 3. Vector Quantization: z_i = argmin_j ||ℓ2(h_i) - ℓ2(v_j)||_2
        normalized_features = F.normalize(patch_features, dim=-1)  # ℓ2 normalize
        normalized_codebook = F.normalize(self.codebook.weight, dim=-1)  # ℓ2 normalize
        
        # Compute distances and find nearest codebook entries
        # [batch, patches, embed_dim] @ [embed_dim, vocab_size] -> [batch, patches, vocab_size]
        similarities = torch.matmul(normalized_features, normalized_codebook.t())
        indices = torch.argmax(similarities, dim=-1)  # [batch, patches]
        
        # Get quantized embeddings
        quantized = self.codebook(indices)  # [batch, patches, embed_dim]
        quantized_normalized = F.normalize(quantized, dim=-1)
        
        if not return_loss:
            return quantized_normalized, indices
        
        # 4. VQ-KD Loss computation
        # Decode quantized features
        decoder_output = self.decoder(
            quantized_normalized,  # Target sequence
            quantized_normalized   # Memory sequence (self-attention)
        )
        
        # Project to teacher dimension
        reconstructed = self.output_proj(decoder_output)  # [batch, patches, teacher_dim]
        reconstructed_normalized = F.normalize(reconstructed, dim=-1)
        teacher_normalized = F.normalize(teacher_features, dim=-1)
        
        # VQ-KD Loss components
        # 1. Cosine similarity loss: maximize cos(o_i, t_i)
        teacher_dim = teacher_features.size(-1)
        cosine_loss = 1 - F.cosine_similarity(
            reconstructed_normalized.view(-1, teacher_dim),
            teacher_normalized.view(-1, teacher_dim),
            dim=-1
        ).mean()
        
        # 2. Commitment losses với stop gradient
        commitment_loss = F.mse_loss(
            normalized_features,  # sg[ℓ2(h_i)]
            quantized_normalized.detach()  # ℓ2(v_{z_i})
        )
        
        codebook_loss = F.mse_loss(
            normalized_features.detach(),  # ℓ2(h_i)
            quantized_normalized  # sg[ℓ2(v_{z_i})]
        )
        
        # Total VQ-KD loss
        total_loss = cosine_loss + commitment_loss + codebook_loss
        
        return quantized_normalized, indices, {
            'vq_kd_loss': total_loss,
            'cosine_loss': cosine_loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss
        }

class MultiwayFusionLayer(nn.Module):
    """
    Enhanced multimodal fusion với Multiway Transformers
    Thay thế ImprovedMultimodalFusionLayer
    """
    
    def __init__(self, vision_dim, text_dim, hidden_dim, num_layers=6, dropout_rate=0.1):
        super().__init__()
        
        assert hidden_dim == text_dim, f"hidden_dim ({hidden_dim}) should match text_dim ({text_dim})"
        
        # Dimension alignment
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Multiway Transformer layers
        # 3 layer cuối sẽ có vision-language expert
        self.multiway_layers = nn.ModuleList([
            MultiWayTransformerLayer(
                hidden_dim, 
                num_heads=8,
                dropout_rate=dropout_rate,
                is_fusion_layer=(i >= num_layers - 3)  # 3 layers cuối có fusion expert
            )
            for i in range(num_layers)
        ])
        
        self.num_layers = num_layers
        print(f"Multiway Fusion: {num_layers} layers, top 3 have vision-language experts")
        
    def forward(self, vision_features, text_features, text_attention_mask=None):
        batch_size = vision_features.size(0)
        vision_patches = vision_features.size(1)
        text_len = text_features.size(1)
        
        # Project features
        vision_proj = self.vision_proj(vision_features)  # [batch, patches, hidden_dim]
        text_proj = self.text_proj(text_features)        # [batch, seq_len, hidden_dim]
        
        # Combine vision và text tokens
        combined_features = torch.cat([vision_proj, text_proj], dim=1)  # [batch, patches+seq_len, hidden_dim]
        
        # Create modality mask
        # 0: vision, 1: language, 2: fusion (để dành cho later use)
        modality_mask = torch.zeros(batch_size, vision_patches + text_len, dtype=torch.long, device=vision_features.device)
        modality_mask[:, :vision_patches] = 0  # Vision tokens
        modality_mask[:, vision_patches:] = 1  # Language tokens
        
        # Create attention mask
        vision_mask = torch.ones(batch_size, vision_patches, device=vision_features.device, dtype=torch.bool)
        if text_attention_mask is not None:
            combined_attention_mask = torch.cat([vision_mask, text_attention_mask.bool()], dim=1)
        else:
            text_mask = torch.ones(batch_size, text_len, device=text_features.device, dtype=torch.bool)
            combined_attention_mask = torch.cat([vision_mask, text_mask], dim=1)
        
        # Pass through Multiway Transformer layers
        x = combined_features
        for i, layer in enumerate(self.multiway_layers):
            x, _ = layer(x, modality_mask, combined_attention_mask)
        
        return x, combined_attention_mask

def apply_block_wise_vision_masking(vision_features, mask_ratio=0.4, block_size=4):
    """
    Apply block-wise masking to vision patches according to BARTPhoBEiT paper
    Args:
        vision_features: [batch, num_patches, dim]
        mask_ratio: Ratio of patches to mask (0.4 = 40%)
        block_size: Size of each mask block
    """
    batch_size, num_patches, dim = vision_features.shape
    
    # ViT has CLS token at position 0, patches start from position 1
    has_cls_token = True  # ViT always has CLS token
    actual_patches = num_patches - 1  # 196 patches (197 - 1 CLS)
    grid_size = int(math.sqrt(actual_patches))  # 14x14 = 196
    
    if grid_size * grid_size != actual_patches:
        print(f"Warning: Cannot form perfect square grid. patches={actual_patches}, grid_size={grid_size}")
        return vision_features, torch.zeros(batch_size, num_patches, device=vision_features.device, dtype=torch.bool)
    
    masked_features = vision_features.clone()
    mask_indicators = torch.zeros(batch_size, num_patches, device=vision_features.device, dtype=torch.bool)
    
    # FIX: Use a more aggressive masking approach
    for batch_idx in range(batch_size):
        # Calculate total patches to mask
        target_masked_patches = int(actual_patches * mask_ratio)
        
        # FIX: Use random patch-based masking if block-based doesn't achieve target
        blocks_per_dim = grid_size // block_size  # 14 // 4 = 3
        max_blocks_per_dim = (grid_size + block_size - 1) // block_size  # Ceiling division = 4
        
        # Try overlapping blocks to achieve higher masking ratio
        all_possible_blocks = []
        for i in range(max_blocks_per_dim):
            for j in range(max_blocks_per_dim):
                start_i = min(i * block_size, grid_size - block_size)
                start_j = min(j * block_size, grid_size - block_size)
                all_possible_blocks.append((start_i, start_j))
        
        # Calculate how many blocks needed for target ratio
        patches_per_block = min(block_size * block_size, actual_patches)
        num_blocks_needed = max(1, (target_masked_patches + patches_per_block - 1) // patches_per_block)
        num_blocks_to_use = min(num_blocks_needed, len(all_possible_blocks))
        
        # Randomly select blocks
        selected_blocks = random.sample(all_possible_blocks, num_blocks_to_use)
        
        total_masked_patches = 0
        masked_patch_set = set()
        
        # Apply masking to selected blocks
        for block_i, block_j in selected_blocks:
            end_i = min(block_i + block_size, grid_size)
            end_j = min(block_j + block_size, grid_size)
            
            # FIX: Zero out the features correctly and track indices
            for pi in range(block_i, end_i):
                for pj in range(block_j, end_j):
                    patch_idx = pi * grid_size + pj
                    if patch_idx not in masked_patch_set and patch_idx < actual_patches:
                        # Convert to tensor index (add 1 for CLS token)
                        tensor_idx = patch_idx + 1
                        
                        # Actually zero out the features
                        masked_features[batch_idx, tensor_idx, :] = 0.0
                        mask_indicators[batch_idx, tensor_idx] = True
                        
                        masked_patch_set.add(patch_idx)
                        total_masked_patches += 1
        
        # FIX: If still not enough, add random patches
        if total_masked_patches < target_masked_patches:
            remaining_patches = target_masked_patches - total_masked_patches
            unmasked_patches = [i for i in range(actual_patches) if i not in masked_patch_set]
            
            if len(unmasked_patches) > 0:
                additional_patches = random.sample(
                    unmasked_patches, 
                    min(remaining_patches, len(unmasked_patches))
                )
                
                for patch_idx in additional_patches:
                    tensor_idx = patch_idx + 1
                    masked_features[batch_idx, tensor_idx, :] = 0.0
                    mask_indicators[batch_idx, tensor_idx] = True
                    total_masked_patches += 1
        
        # Enhanced debug info
        actual_mask_ratio = total_masked_patches / actual_patches
        # print(f"  Batch {batch_idx}: {len(selected_blocks)} blocks selected, "
        #       f"{total_masked_patches}/{actual_patches} patches ({actual_mask_ratio:.3f} ratio)")
        # print(f"    Target: {target_masked_patches} patches ({mask_ratio:.3f} ratio)")
        
        # Verify masking worked by checking some masked patches
        if total_masked_patches > 0:
            sample_masked_idx = list(masked_patch_set)[:3]  # Check first 3 masked patches
            for idx in sample_masked_idx:
                tensor_idx = idx + 1
                patch_norm = masked_features[batch_idx, tensor_idx, :].norm().item()
                # print(f"    Masked patch {idx} (tensor idx {tensor_idx}): norm = {patch_norm:.6f}")
    
    return masked_features, mask_indicators


class ImprovedVietnameseVQAModel(nn.Module):
    """Enhanced Vietnamese VQA Model với Full BART và Block-wise Vision Masking"""
    
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        
        # Vision encoder (ViT)
        self.vision_model = ViTModel.from_pretrained(model_config['vision_model'])
        vision_dim = self.vision_model.config.hidden_size  # 768
        
        # Load BART model only once
        bart_model = MBartForConditionalGeneration.from_pretrained(model_config['text_model'])
        text_dim = bart_model.config.d_model  # 1024 for BARTPho
        
        # Use BART encoder for text encoding
        self.text_encoder = bart_model.get_encoder()
        
        self.text_decoder = bart_model  # Reuse the same model
        
        # Tokenizer
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(model_config['decoder_model'])

        print(f"Using full BART model: {model_config['text_model']}")
        print(f"  Vision model: {model_config['vision_model']} (dim: {vision_dim})")
        print(f"  Text model: {model_config['text_model']} (dim: {text_dim})")
        print(f"  Decoder model: {model_config['decoder_model']} (dim: {text_dim})")

        # VQ-KD Visual Tokenizer
        use_vqkd = model_config.get('use_vqkd', True)
        if use_vqkd:
            self.visual_tokenizer = VQKDVisualTokenizer(
                vocab_size=model_config.get('visual_vocab_size', 8192),
                embed_dim=vision_dim,
                teacher_dim=vision_dim
            )
            print("VQ-KD Visual Tokenizer enabled")
        else:
            self.visual_tokenizer = None
            print("VQ-KD Visual Tokenizer disabled")
        
        # Enhanced fusion layer với Multiway Transformers
        hidden_dim = model_config.get('hidden_dim', 1024)
        dropout_rate = model_config.get('dropout_rate', 0.1)
        num_multiway_layers = model_config.get('num_multiway_layers', 6)
        
        assert hidden_dim == text_dim, f"hidden_dim ({hidden_dim}) must match BART dim ({text_dim})"
        
        self.fusion_layer = MultiwayFusionLayer(
            vision_dim, text_dim, hidden_dim, num_multiway_layers, dropout_rate
        )
        
        # Output projection to BART decoder dimension
        decoder_dim = self.text_decoder.config.d_model
        assert decoder_dim == hidden_dim, f"Decoder dim ({decoder_dim}) must match hidden dim ({hidden_dim})"
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(decoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim)
        )
        
        # Block-wise masking configuration
        self.use_masking = model_config.get('use_unified_masking', False)
        self.vision_mask_ratio = model_config.get('vision_mask_ratio', 0.4)
        self.vision_mask_block_size = model_config.get('vision_mask_block_size', 4)
        self.text_mask_ratio = model_config.get('text_mask_ratio', 0.15)
        self.multimodal_text_mask_ratio = model_config.get('multimodal_text_mask_ratio', 0.5)
        
        # Label smoothing loss
        self.label_smoothing = model_config.get('label_smoothing', 0.1)
        
        # Initialize with frozen encoders
        self.freeze_encoders()
        
        # Pool target length
        self.pool_target_length = model_config.get('pool_target_length', 32)
        self.pool_queries = nn.Parameter(torch.randn(self.pool_target_length, decoder_dim))
        self.pool_attn = nn.MultiheadAttention(decoder_dim, num_heads=8, batch_first=True)
        self.pool_pos_emb = nn.Embedding(self.pool_target_length, decoder_dim)
        
        # Initialize
        nn.init.normal_(self.pool_pos_emb.weight, std=0.02)
        nn.init.normal_(self.pool_queries, std=0.02)
        
        # Debug
        self._debug_step = 0
        self._debug_freq = 100

        print(f"Enhanced BARTPhoBEiT Model with Full BART:")
        print(f"  Vision model: {model_config['vision_model']} (dim: {vision_dim})")
        print(f"  Text model: {model_config['text_model']} (dim: {text_dim})")
        print(f"  Decoder model: {model_config['decoder_model']} (dim: {decoder_dim})")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Multiway layers: {num_multiway_layers}")
        print(f"  VQ-KD enabled: {use_vqkd}")
        print(f"  Block-wise masking: {self.use_masking} (ratio: {self.vision_mask_ratio}, block_size: {self.vision_mask_block_size})")
        print(f"  Label smoothing: {self.label_smoothing}")
    
    def apply_text_masking(self, tokens, mask_ratio=0.15):
        """Apply text masking strategy"""
        if not self.training or not self.use_masking:
            return tokens, None
        
        batch_size, seq_len = tokens.shape
        
        # Random masking for text
        mask = torch.rand(batch_size, seq_len, device=tokens.device) < mask_ratio
        
        masked_tokens = tokens.clone()
        # Replace with mask token for text
        mask_token_id = self.decoder_tokenizer.mask_token_id or self.decoder_tokenizer.pad_token_id
        masked_tokens[mask] = mask_token_id
        
        return masked_tokens, mask
    
    def freeze_encoders(self):
        """Freeze all encoder parameters"""
        frozen_params = 0
        
        for param in self.vision_model.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            
        print(f"Encoders frozen ({frozen_params:,} parameters)")
    
    def partial_unfreeze(self, unfreeze_last_n_layers=2):
        """Enhanced partial unfreezing with better layer detection"""
        unfrozen_params = 0
        
        # BART encoder unfreezing
        print(f"\nBART encoder structure analysis:")
        bart_layers = None
        
        if hasattr(self.text_encoder, 'layers'):
            bart_layers = self.text_encoder.layers
            print(f"  Found encoder.layers with {len(bart_layers)} layers")
        
        if bart_layers:
            for i, layer in enumerate(bart_layers[-unfreeze_last_n_layers:], len(bart_layers)-unfreeze_last_n_layers):
                layer_params = sum(p.numel() for p in layer.parameters())
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                print(f"    Unfrozen BART encoder layer {i}: {layer_params:,} params")
        
        # ViT unfreezing
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
            for i, layer in enumerate(vit_layers[-unfreeze_last_n_layers:], len(vit_layers)-unfreeze_last_n_layers):
                layer_params = sum(p.numel() for p in layer.parameters())
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                print(f"    Unfrozen ViT layer {i}: {layer_params:,} params")
        
        # Always unfreeze layer norms and embeddings
        norm_params = 0
        for name, param in self.text_encoder.named_parameters():
            if any(keyword in name.lower() for keyword in ['layernorm', 'layer_norm', 'embed']):
                if not param.requires_grad:
                    norm_params += param.numel()
                param.requires_grad = True
        
        for name, param in self.vision_model.named_parameters():
            if any(keyword in name.lower() for keyword in ['layernorm', 'layer_norm', 'embed']):
                if not param.requires_grad:
                    norm_params += param.numel()
                param.requires_grad = True
        
        unfrozen_params += norm_params
        
        print(f"\nPartial unfreezing summary:")
        print(f"  Total unfrozen parameters: {unfrozen_params:,}")
        print(f"  Last {unfreeze_last_n_layers} layers unfrozen successfully")
    
    def create_encoder_outputs(self, encoder_states):
        """Create proper encoder outputs for BART decoder"""
        return BaseModelOutput(
            last_hidden_state=encoder_states,
            hidden_states=None,
            attentions=None
        )
    
    def forward(self, pixel_values, question_input_ids, question_attention_mask, 
                answer_input_ids=None, answer_attention_mask=None, is_multimodal=True):
        
        batch_size = pixel_values.size(0)
        self._debug_step += 1
        
        # Encode image với VQ-KD nếu có
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [batch, patches, 768]
        
        vq_losses = {}
        if self.visual_tokenizer is not None and self.training:
            # Apply VQ-KD visual tokenization
            quantized_vision, visual_indices, vq_loss_dict = self.visual_tokenizer(
                pixel_values, return_loss=True
            )
            # Use quantized features for downstream processing
            vision_features = quantized_vision
            vq_losses = vq_loss_dict
        
        # Apply block-wise vision masking
        if self.use_masking and self.training:
            vision_features, vision_mask = apply_block_wise_vision_masking(
                vision_features, 
                mask_ratio=self.vision_mask_ratio,
                block_size=self.vision_mask_block_size
            )
        
        # Encode question using BART encoder
        text_outputs = self.text_encoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, 1024]
        
        # Apply text masking
        if self.use_masking and self.training:
            mask_ratio = self.multimodal_text_mask_ratio if is_multimodal else self.text_mask_ratio
            masked_question_ids, text_mask = self.apply_text_masking(
                question_input_ids, mask_ratio
            )
            # Re-encode với masked tokens
            masked_text_outputs = self.text_encoder(
                input_ids=masked_question_ids,
                attention_mask=question_attention_mask
            )
            text_features = masked_text_outputs.last_hidden_state
        
        # Multiway Transformer fusion
        fused_features, combined_attention_mask = self.fusion_layer(
            vision_features, text_features, question_attention_mask
        )
        
        # Project features for BART decoder
        encoder_states = self.output_proj(fused_features)
        
        # Debug logging
        if self.training and self._debug_step % self._debug_freq == 0:
            print(f"Debug shapes (step {self._debug_step}):")
            print(f"  Vision features: {vision_features.shape}")
            print(f"  Text features: {text_features.shape}")
            print(f"  Fused features: {fused_features.shape}")
            print(f"  Encoder states: {encoder_states.shape}")
            if vq_losses:
                print(f"  VQ-KD loss: {vq_losses.get('vq_kd_loss', 0):.4f}")
        
        if answer_input_ids is not None:  # Training mode
            # Validate token IDs
            vocab_size = self.text_decoder.config.vocab_size
            invalid_tokens = (answer_input_ids >= vocab_size) | (answer_input_ids < 0)
            if invalid_tokens.any():
                answer_input_ids = torch.clamp(answer_input_ids, 0, vocab_size - 1)
            
            decoder_seq_len = min(answer_input_ids.size(1), self.pool_target_length)
            pooled_encoder_states = self.create_pooled_representation(
                encoder_states, combined_attention_mask, target_length=decoder_seq_len
            )
            
            encoder_outputs = self.create_encoder_outputs(pooled_encoder_states)
            
            if self.label_smoothing > 0:
                # Custom loss with label smoothing + VQ-KD loss
                decoder_outputs = self.text_decoder(
                    input_ids=answer_input_ids[:, :decoder_seq_len],
                    attention_mask=answer_attention_mask[:, :decoder_seq_len] if answer_attention_mask is not None else None,
                    encoder_outputs=encoder_outputs
                )
                
                logits = decoder_outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = answer_input_ids[:, 1:decoder_seq_len].contiguous()
                
                # Main generation loss
                generation_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.decoder_tokenizer.pad_token_id,
                    label_smoothing=self.label_smoothing
                )
                
                # Total loss = generation loss + VQ-KD loss
                total_loss = generation_loss
                if vq_losses and 'vq_kd_loss' in vq_losses:
                    total_loss = total_loss + 0.1 * vq_losses['vq_kd_loss']  # Weight VQ-KD loss
                
                # Calculate token accuracy
                with torch.no_grad():
                    predictions = torch.argmax(shift_logits, dim=-1)
                    mask = (shift_labels != self.decoder_tokenizer.pad_token_id)
                    correct_predictions = (predictions == shift_labels) & mask
                    token_accuracy = correct_predictions.sum().float() / mask.sum().float() if mask.sum() > 0 else 0.0
                
                if self.training and self._debug_step % self._debug_freq == 0:
                    print(f"  Generation loss: {generation_loss.item():.4f}")
                    print(f"  Total loss: {total_loss.item():.4f}")
                    print(f"  Token accuracy: {token_accuracy:.4f}")
                
                class CustomDecoderOutput:
                    def __init__(self, loss, logits, token_accuracy=None, vq_losses=None):
                        self.loss = loss
                        self.logits = logits
                        self.token_accuracy = token_accuracy
                        self.vq_losses = vq_losses or {}
                
                return CustomDecoderOutput(total_loss, logits, token_accuracy, vq_losses)
            
            else:
                # Standard loss + VQ-KD loss
                decoder_outputs = self.text_decoder(
                    input_ids=answer_input_ids[:, :decoder_seq_len],
                    attention_mask=answer_attention_mask[:, :decoder_seq_len] if answer_attention_mask is not None else None,
                    encoder_outputs=encoder_outputs,
                    labels=answer_input_ids[:, :decoder_seq_len]
                )
                
                # Add VQ-KD loss
                if vq_losses and 'vq_kd_loss' in vq_losses:
                    decoder_outputs.loss = decoder_outputs.loss + 0.1 * vq_losses['vq_kd_loss']
                
                return decoder_outputs
            
        else:  # Inference mode
            pooled_encoder_states = self.create_pooled_representation(
                encoder_states, combined_attention_mask, target_length=self.pool_target_length
            )
            
            encoder_outputs = self.create_encoder_outputs(pooled_encoder_states)
            
            generated_ids = self.text_decoder.generate(
                encoder_outputs=encoder_outputs,
                max_length=self.pool_target_length,
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
        batch_size, seq_len, hidden_dim = encoder_states.shape
        if target_length is None:
            target_length = self.pool_target_length

        # Attention pooling with learnable queries
        queries = self.pool_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add positional embeddings
        pos_ids = torch.arange(target_length, device=queries.device)
        pos_emb = self.pool_pos_emb(pos_ids).unsqueeze(0)
        queries = queries + pos_emb
        
        # Create key padding mask
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        # Apply attention pooling
        pooled, attention_weights = self.pool_attn(
            queries, encoder_states, encoder_states,
            key_padding_mask=key_padding_mask
        )
        
        return pooled

# WUPS Metrics Implementation
def compute_wups(predicted_answers, reference_answers_list, threshold=0.0):
    """
    Compute WUPS (Word-level Unigram Precision-based Semantic) score
    Args:
        predicted_answers: List of predicted answer strings
        reference_answers_list: List of lists of reference answer strings
        threshold: WUPS threshold (0.0 or 0.9)
    """
    try:
        from nltk.corpus import wordnet
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except ImportError:
        print("Warning: NLTK not available for WUPS computation")
        return 0.0
    
    def get_wordnet_similarity(word1, word2):
        """Get WordNet-based similarity between two words"""
        if word1 == word2:
            return 1.0
        
        synsets1 = wordnet.synsets(word1)
        synsets2 = wordnet.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        max_similarity = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                similarity = syn1.path_similarity(syn2)
                if similarity and similarity > max_similarity:
                    max_similarity = similarity
        
        return max_similarity if max_similarity is not None else 0.0
    
    def compute_wups_single(pred_tokens, ref_tokens_list, threshold):
        """Compute WUPS for single prediction against multiple references"""
        best_wups = 0.0
        
        for ref_tokens in ref_tokens_list:
            if not pred_tokens and not ref_tokens:
                wups_score = 1.0
            elif not pred_tokens or not ref_tokens:
                wups_score = 0.0
            else:
                # Compute precision and recall
                precision_scores = []
                for pred_word in pred_tokens:
                    max_sim = 0.0
                    for ref_word in ref_tokens:
                        sim = get_wordnet_similarity(pred_word.lower(), ref_word.lower())
                        if sim >= threshold:
                            max_sim = max(max_sim, sim)
                    precision_scores.append(max_sim)
                
                recall_scores = []
                for ref_word in ref_tokens:
                    max_sim = 0.0
                    for pred_word in pred_tokens:
                        sim = get_wordnet_similarity(ref_word.lower(), pred_word.lower())
                        if sim >= threshold:
                            max_sim = max(max_sim, sim)
                    recall_scores.append(max_sim)
                
                precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
                recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
                
                if precision + recall > 0:
                    wups_score = 2 * precision * recall / (precision + recall)
                else:
                    wups_score = 0.0
            
            best_wups = max(best_wups, wups_score)
        
        return best_wups
    
    wups_scores = []
    
    for pred_answer, ref_answers in zip(predicted_answers, reference_answers_list):
        # Normalize and tokenize
        pred_normalized = normalize_vietnamese_answer(pred_answer)
        pred_tokens = pred_normalized.split()
        
        ref_tokens_list = []
        for ref_answer in ref_answers:
            ref_normalized = normalize_vietnamese_answer(ref_answer)
            ref_tokens_list.append(ref_normalized.split())
        
        wups_score = compute_wups_single(pred_tokens, ref_tokens_list, threshold)
        wups_scores.append(wups_score)
    
    return sum(wups_scores) / len(wups_scores) if wups_scores else 0.0

# Enhanced metrics computation with WUPS
def compute_metrics_with_multiple_answers(predictions, all_correct_answers_list, tokenizer=None):
    """Enhanced evaluation metrics supporting multiple correct answers with VQA score and WUPS"""
    
    norm_preds = [normalize_vietnamese_answer(pred) for pred in predictions]
    
    metrics = {}
    
    # VQA Score
    vqa_metrics = compute_vqa_score_batch(predictions, all_correct_answers_list)
    metrics.update(vqa_metrics)
    
    # WUPS metrics
    try:
        metrics['wups_0.0'] = compute_wups(predictions, all_correct_answers_list, threshold=0.0)
        metrics['wups_0.9'] = compute_wups(predictions, all_correct_answers_list, threshold=0.9)
        print(f"✓ WUPS metrics calculated successfully")
    except Exception as e:
        print(f"Warning: WUPS calculation failed: {e}")
        metrics['wups_0.0'] = 0.0
        metrics['wups_0.9'] = 0.0
    
    # Multi-answer exact match accuracy
    multi_exact_matches = []
    for pred, correct_answers in zip(norm_preds, all_correct_answers_list):
        norm_correct_answers = [normalize_vietnamese_answer(ans) for ans in correct_answers]
        is_exact_match = pred in norm_correct_answers
        multi_exact_matches.append(is_exact_match)
    
    metrics['multi_exact_accuracy'] = sum(multi_exact_matches) / len(multi_exact_matches)
    
    # Best fuzzy accuracy
    best_fuzzy_scores = []
    for pred, correct_answers in zip(norm_preds, all_correct_answers_list):
        norm_correct_answers = [normalize_vietnamese_answer(ans) for ans in correct_answers]
        
        fuzzy_scores = []
        for ref in norm_correct_answers:
            if pred == ref:
                fuzzy_scores.append(1.0)
            elif pred in ref or ref in pred:
                fuzzy_scores.append(0.8)
            else:
                similarity = SequenceMatcher(None, pred, ref).ratio()
                fuzzy_scores.append(similarity)
        
        best_score = max(fuzzy_scores) if fuzzy_scores else 0.0
        best_fuzzy_scores.append(best_score)
    
    metrics['multi_fuzzy_accuracy'] = sum(best_fuzzy_scores) / len(best_fuzzy_scores)
    
    # Token-level F1 score
    best_token_f1_scores = []
    best_token_precisions = []
    best_token_recalls = []
    
    for pred, correct_answers in zip(norm_preds, all_correct_answers_list):
        norm_correct_answers = [normalize_vietnamese_answer(ans) for ans in correct_answers]
        pred_tokens = set(pred.split())
        
        f1_scores = []
        precisions = []
        recalls = []
        
        for ref in norm_correct_answers:
            ref_tokens = set(ref.split())
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                f1_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            elif len(pred_tokens) == 0:
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                common_tokens = pred_tokens.intersection(ref_tokens)
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
        
        best_token_precisions.append(max(precisions) if precisions else 0.0)
        best_token_recalls.append(max(recalls) if recalls else 0.0)
        best_token_f1_scores.append(max(f1_scores) if f1_scores else 0.0)
    
    metrics['multi_token_precision'] = sum(best_token_precisions) / len(best_token_precisions)
    metrics['multi_token_recall'] = sum(best_token_recalls) / len(best_token_recalls)
    metrics['multi_token_f1'] = sum(best_token_f1_scores) / len(best_token_f1_scores)
    
    # BLEU score với multiple references
    bleu_scores = []
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothing = SmoothingFunction().method4
        
        for pred, correct_answers in zip(norm_preds, all_correct_answers_list):
            norm_correct_answers = [normalize_vietnamese_answer(ans) for ans in correct_answers]
            pred_tokens = pred.split()
            ref_tokens_list = [ref.split() for ref in norm_correct_answers]
            
            try:
                bleu = sentence_bleu(ref_tokens_list, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
        
        metrics['multi_bleu'] = sum(bleu_scores) / len(bleu_scores)
        
    except ImportError:
        metrics['multi_bleu'] = 0.0
    
    # ROUGE-L score với multiple references
    rouge_scores = []
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        
        for pred, correct_answers in zip(norm_preds, all_correct_answers_list):
            norm_correct_answers = [normalize_vietnamese_answer(ans) for ans in correct_answers]
            
            rouge_scores_for_pred = []
            for ref in norm_correct_answers:
                try:
                    score = scorer.score(ref, pred)['rougeL'].fmeasure
                    rouge_scores_for_pred.append(score)
                except:
                    rouge_scores_for_pred.append(0.0)
            
            best_rouge = max(rouge_scores_for_pred) if rouge_scores_for_pred else 0.0
            rouge_scores.append(best_rouge)
        
        metrics['multi_rouge_l'] = sum(rouge_scores) / len(rouge_scores)
        
    except ImportError:
        metrics['multi_rouge_l'] = 0.0
    
    # Enhanced VQA score analysis
    vqa_score_distribution = {
        'perfect_ratio': metrics['vqa_perfect_count'] / len(predictions),
        'zero_ratio': metrics['vqa_zero_count'] / len(predictions),
        'partial_ratio': metrics['vqa_partial_count'] / len(predictions)
    }
    metrics.update(vqa_score_distribution)
    
    # FIX: Remove recursive call - compute traditional single-answer metrics directly
    first_answers = [correct_answers[0] if correct_answers else "" for correct_answers in all_correct_answers_list]
    
    # Direct traditional metrics computation (no recursion)
    norm_first_refs = [normalize_vietnamese_answer(ref) for ref in first_answers]
    
    exact_matches = [pred == ref for pred, ref in zip(norm_preds, norm_first_refs)]
    metrics['single_exact_accuracy'] = sum(exact_matches) / len(exact_matches)
    
    fuzzy_scores = []
    for pred, ref in zip(norm_preds, norm_first_refs):
        if pred == ref:
            fuzzy_scores.append(1.0)
        elif pred in ref or ref in pred:
            fuzzy_scores.append(0.8)
        else:
            similarity = SequenceMatcher(None, pred, ref).ratio()
            fuzzy_scores.append(similarity)
    
    metrics['single_fuzzy_accuracy'] = sum(fuzzy_scores) / len(fuzzy_scores)
    
    metrics['total_samples'] = len(predictions)
    metrics['multi_exact_matches'] = sum(multi_exact_matches)
    
    return metrics

def compute_vqa_score_single(prediction, reference_answers):
    """
    Compute VQA score for single prediction against multiple reference answers
    VQA Score = min(# humans that provided that answer / 3, 1.0)
    """
    if not reference_answers:
        return 0.0
    
    # FIX: Handle None/empty predictions
    if not prediction:
        prediction = ""
    
    norm_pred = normalize_vietnamese_answer(prediction)
    
    # FIX: Safe normalization with error handling
    norm_refs = []
    for ref in reference_answers:
        try:
            if ref is not None:
                norm_ref = normalize_vietnamese_answer(ref)
                norm_refs.append(norm_ref)
        except Exception as e:
            print(f"Warning: Error normalizing reference '{ref}': {e}")
            norm_refs.append(str(ref).lower().strip() if ref else "")
    
    if not norm_refs:
        return 0.0
    
    match_count = norm_refs.count(norm_pred)
    vqa_score = min(match_count / 3.0, 1.0)
    
    return vqa_score

def compute_vqa_score_batch(predictions, all_reference_answers_list):
    """
    Compute VQA scores for batch of predictions
    """
    vqa_scores = []
    
    for pred, ref_answers in zip(predictions, all_reference_answers_list):
        vqa_score = compute_vqa_score_single(pred, ref_answers)
        vqa_scores.append(vqa_score)
    
    metrics = {
        'vqa_score': sum(vqa_scores) / len(vqa_scores) if vqa_scores else 0.0,
        'vqa_scores_list': vqa_scores,
        'vqa_perfect_count': sum(1 for score in vqa_scores if score == 1.0),
        'vqa_zero_count': sum(1 for score in vqa_scores if score == 0.0),
        'vqa_partial_count': sum(1 for score in vqa_scores if 0.0 < score < 1.0)
    }
    
    return metrics

def compute_metrics(predictions, references_or_multi_answers, tokenizer=None):
    """Enhanced compute_metrics that automatically detects multiple answers and includes WUPS"""
    
    if len(references_or_multi_answers) > 0 and isinstance(references_or_multi_answers[0], list):
        return compute_metrics_with_multiple_answers(predictions, references_or_multi_answers, tokenizer)
    else:
        single_answers_as_lists = [[ref] for ref in references_or_multi_answers]
        # FIX: Call directly without recursion
        return compute_metrics_with_multiple_answers(predictions, single_answers_as_lists, tokenizer)

# Data augmentation functions (unchanged)
def augment_question(question, augment_ratio=0.2):
    """Simple Vietnamese question augmentation"""
    if random.random() > augment_ratio:
        return question
    
    words = question.split()
    if len(words) < 3:
        return question
    
    if len(words) > 4:
        middle = words[1:-1]
        random.shuffle(middle)
        return ' '.join([words[0]] + middle + [words[-1]])
    
    return question

def normalize_vietnamese_answer(answer):
    """Enhanced Vietnamese answer normalization"""
    if not isinstance(answer, str):
        answer = str(answer)
    
    # FIX: Handle None/empty cases
    if not answer or answer.strip() == '':
        return ''
    
    try:
        answer = unicodedata.normalize('NFC', answer)
        answer = answer.lower().strip()
        
        # FIX: Escape regex pattern properly
        answer = re.sub(r'[.,!?;:"\'()\[\]{}]', '', answer)  # Escape square brackets
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        vietnamese_stopwords = ['các', 'của', 'và', 'là', 'trong', 'với', 'để', 'được', 'một', 'này', 'đó']
        words = answer.split()
        filtered_words = [w for w in words if w not in vietnamese_stopwords or len(words) <= 2]
        
        return ' '.join(filtered_words) if filtered_words else answer
        
    except Exception as e:
        print(f"Warning: Error normalizing answer '{answer}': {e}")
        return str(answer).lower().strip()