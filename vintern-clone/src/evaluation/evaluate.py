# src/evaluation/evaluate.py
import torch
from transformers import AutoTokenizer, AutoImageProcessor
from src.model.vintern_model import VinternModel
from PIL import Image
import json

class VinternEvaluator:
    def __init__(self, model_path):
        self.model = VinternModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        self.image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/InternViT-300M-448px")
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
    
    def generate_answer(self, image_path, question, max_length=512):
        """Generate answer for a given image and question"""
        
        # Process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.process_image(image).unsqueeze(0)
        
        # Process text
        prompt = f"USER: {question} ASSISTANT:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                images=image_tensor,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip()
        
        return answer
    
    def process_image(self, image):
        """Process image similar to training"""
        # Implement same image processing logic as in dataset
        pass
    
    def evaluate_dataset(self, test_data_path):
        """Evaluate on test dataset"""
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        for item in test_data:
            question = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            
            predicted_answer = self.generate_answer(item['image_path'], question)
            
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted_answer,
                'image_path': item['image_path']
            })
        
        return results

def main():
    evaluator = VinternEvaluator("models/checkpoints/final_model")
    results = evaluator.evaluate_dataset("data/processed/test.json")
    
    # Save results
    with open("outputs/evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()