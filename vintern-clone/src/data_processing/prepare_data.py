# src/data_processing/prepare_data.py
import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class VinternDataset(Dataset):
    def __init__(self, data_path, image_processor, tokenizer, max_length=4096):
        self.data = self.load_data(data_path)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def process_image(self, image_path):
        """Xử lý ảnh theo dynamic high resolution"""
        image = Image.open(image_path).convert('RGB')
        
        # Chia ảnh thành tiles 448x448
        width, height = image.size
        tile_size = 448
        
        # Tính số tiles cần thiết
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        
        tiles = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                left = x * tile_size
                top = y * tile_size
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)
                
                tile = image.crop((left, top, right, bottom))
                tile = tile.resize((tile_size, tile_size))
                tiles.append(self.image_processor(tile))
        
        return torch.stack(tiles)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Xử lý ảnh
        image_tensor = self.process_image(item['image_path'])
        
        # Xử lý text
        conversations = item['conversations']
        text = self.format_conversations(conversations)
        
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'image': image_tensor,
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze()
        }
    
    def format_conversations(self, conversations):
        """Format conversations theo style LLaVA"""
        formatted = ""
        for conv in conversations:
            if conv['from'] == 'human':
                formatted += f"USER: {conv['value']} "
            else:
                formatted += f"ASSISTANT: {conv['value']}"
        return formatted