# src/model/vintern_model.py
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, 
    AutoImageProcessor, AutoModelForCausalLM
)
from peft import get_peft_model, LoraConfig

class VinternModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Vision Encoder - InternViT-300M-448px
        self.vision_encoder = AutoModel.from_pretrained(
            "OpenGVLab/InternViT-300M-448px"
        )
        
        # Language Model - Qwen2-0.5B-Instruct  
        self.language_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-0.5B-Instruct"
        )
        
        # MLP Projector
        vision_hidden_size = self.vision_encoder.config.hidden_size
        text_hidden_size = self.language_model.config.hidden_size
        
        self.mlp_projector = nn.Sequential(
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size)
        )
        
        # Pixel shuffle để giảm số tokens
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # Setup LoRA cho language model
        if config.use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """Thiết lập LoRA cho fine-tuning hiệu quả"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
    
    def encode_images(self, images):
        """Encode ảnh thành visual features"""
        batch_size, num_tiles, channels, height, width = images.shape
        
        # Reshape để xử lý tất cả tiles cùng lúc
        images = images.view(-1, channels, height, width)
        
        # Encode qua vision encoder
        with torch.no_grad():
            vision_outputs = self.vision_encoder(images)
            visual_features = vision_outputs.last_hidden_state
        
        # Apply pixel shuffle để giảm tokens (448x448 -> 256 tokens)
        visual_features = self.pixel_shuffle(visual_features.permute(0, 2, 1))
        visual_features = visual_features.permute(0, 2, 1)
        
        # Project to text space
        visual_embeds = self.mlp_projector(visual_features)
        
        # Reshape back
        visual_embeds = visual_embeds.view(batch_size, num_tiles, -1, visual_embeds.size(-1))
        
        return visual_embeds
    
    def forward(self, images, input_ids, attention_mask, labels=None):
        # Encode images
        visual_embeds = self.encode_images(images)
        
        # Get text embeddings
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Concatenate visual and text embeddings
        # (Simplified - cần logic phức tạp hơn để insert visual tokens đúng vị trí)
        batch_size = visual_embeds.size(0)
        visual_tokens = visual_embeds.mean(dim=1).mean(dim=1)  # Average pooling
        
        # Expand visual tokens
        visual_tokens = visual_tokens.unsqueeze(1).expand(-1, text_embeds.size(1), -1)
        
        # Combine embeddings (simplified approach)
        combined_embeds = text_embeds + visual_tokens * 0.1
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs