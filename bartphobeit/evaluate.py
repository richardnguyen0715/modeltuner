import os
from BARTphoBEIT import *
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTFeatureExtractor
import json

# Import OpenViVQA evaluation metrics
from OpenViVQA.evaluation.accuracy import Accuracy
from OpenViVQA.evaluation.bleu import Bleu
from OpenViVQA.evaluation.meteor import Meteor  
from OpenViVQA.evaluation.rouge import Rouge
from OpenViVQA.evaluation.cider import Cider
from OpenViVQA.evaluation.precision import Precision
from OpenViVQA.evaluation.recall import Recall
from OpenViVQA.evaluation.f1 import F1

class ComprehensiveVQAEvaluator:
    """Comprehensive VQA evaluator using OpenViVQA metrics"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Initialize all evaluation metrics
        self.metrics = {
            'accuracy': Accuracy(),
            'bleu_1': Bleu(n=1),
            'bleu_2': Bleu(n=2), 
            'bleu_3': Bleu(n=3),
            'bleu_4': Bleu(n=4),
            'meteor': Meteor(),
            'rouge': Rouge(),
            'cider': Cider(),
            'precision': Precision(),
            'recall': Recall(), 
            'f1': F1()
        }
        
    def normalize_answer(self, answer):
        """Normalize answer text for evaluation"""
        import re
        import string
        
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove punctuation
        answer = answer.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespaces
        answer = ' '.join(answer.split())
        
        return answer
    
    def prepare_for_evaluation(self, predictions, ground_truths):
        """Prepare predictions and ground truths in format required by OpenViVQA"""
        
        # OpenViVQA expects format: {question_id: [answers]} for ground truth
        # and {question_id: answer} for predictions
        
        gts = {}
        res = {}
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            question_id = str(i)
            
            # Normalize answers
            pred_normalized = self.normalize_answer(pred)
            gt_normalized = self.normalize_answer(gt)
            
            # For ground truth, OpenViVQA expects a list of possible answers
            gts[question_id] = [gt_normalized]
            res[question_id] = [pred_normalized]  # Some metrics expect list format
            
        return res, gts
    
    def calculate_comprehensive_metrics(self, predictions, ground_truths):
        """Calculate all VQA metrics"""
        
        print("Preparing data for evaluation...")
        res, gts = self.prepare_for_evaluation(predictions, ground_truths)
        
        results = {}
        
        print("Calculating metrics...")
        
        # Calculate each metric
        for metric_name, metric_evaluator in self.metrics.items():
            try:
                print(f"Computing {metric_name}...")
                
                if metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                    # These metrics typically work with individual answers
                    score, scores = metric_evaluator.compute_score(gts, res)
                else:
                    # BLEU, METEOR, ROUGE, CIDEr work with text generation
                    score, scores = metric_evaluator.compute_score(gts, res)
                
                if isinstance(score, (list, tuple)):
                    results[metric_name] = float(score[0]) if len(score) > 0 else 0.0
                else:
                    results[metric_name] = float(score)
                    
                print(f"{metric_name}: {results[metric_name]:.4f}")
                
            except Exception as e:
                print(f"Error computing {metric_name}: {e}")
                results[metric_name] = 0.0
        
        # Add some additional custom metrics
        results['exact_match'] = self.calculate_exact_match(predictions, ground_truths)
        results['total_samples'] = len(predictions)
        results['empty_predictions'] = sum(1 for pred in predictions if not pred.strip())
        
        return results
    
    def calculate_exact_match(self, predictions, ground_truths):
        """Calculate exact match accuracy"""
        exact_matches = 0
        for pred, gt in zip(predictions, ground_truths):
            if self.normalize_answer(pred) == self.normalize_answer(gt):
                exact_matches += 1
        return exact_matches / len(predictions)

def evaluate_model(model_path, test_questions, config, load_pretrained=True):
    """Evaluate trained model with comprehensive metrics"""
    
    # Load model
    model = VietnameseVQAModel(config)
    
    # Load pretrained weights if specified
    if load_pretrained and model_path and os.path.exists(model_path):
        print(f"Loading finetuned model from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} (this is normal for new model components)")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)} (this is normal for model updates)")
                
            print("Model weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Continuing with base model...")
    else:
        print("Using base model (no finetuned weights loaded)")
    
    model = model.to(config['device'])
    model.eval()
    
    # Prepare test dataset
    question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    answer_tokenizer = AutoTokenizer.from_pretrained(config['decoder_model'])
    feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
    
    test_dataset = VietnameseVQADataset(
        test_questions, config['image_dir'], question_tokenizer, answer_tokenizer, feature_extractor
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize comprehensive evaluator
    evaluator = ComprehensiveVQAEvaluator(answer_tokenizer)
    
    predictions = []
    ground_truths = []
    sample_questions = []
    sample_images = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config['device'])
            
            try:
                # Generate predictions
                generated_ids = model(
                    pixel_values=batch['pixel_values'],
                    question_input_ids=batch['question_input_ids'],
                    question_attention_mask=batch['question_attention_mask']
                )
                
                # Decode predictions
                batch_predictions = answer_tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                predictions.extend(batch_predictions)
                ground_truths.extend(batch['answer_text'])
                sample_questions.extend(batch['question_text'])
                
                # Store some sample images for analysis
                if batch_idx < 3:  # Store first few batches
                    sample_images.extend(batch.get('image_name', [f'batch_{batch_idx}'] * len(batch_predictions)))
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                # Add empty predictions to maintain consistency
                batch_size = len(batch['answer_text'])
                predictions.extend([""] * batch_size)
                ground_truths.extend(batch['answer_text'])
                sample_questions.extend(batch['question_text'])
    
    print(f"\nCompleted evaluation on {len(predictions)} samples")
    
    # Calculate comprehensive metrics
    metrics = evaluator.calculate_comprehensive_metrics(predictions, ground_truths)
    
    # Print results
    print_evaluation_results(metrics, load_pretrained and model_path and os.path.exists(model_path))
    
    # Print sample results
    print_sample_results(sample_questions, predictions, ground_truths, sample_images)
    
    return metrics, predictions, ground_truths

def print_evaluation_results(metrics, is_finetuned):
    """Print comprehensive evaluation results"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VQA EVALUATION RESULTS")
    print("="*80)
    
    print(f"Model type: {'Finetuned' if is_finetuned else 'Base (no finetuning)'}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Empty predictions: {metrics['empty_predictions']} ({metrics['empty_predictions']/metrics['total_samples']*100:.2f}%)")
    
    print("\n--- CORE METRICS ---")
    print(f"Exact Match Accuracy: {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.2f}%)")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    print("\n--- TEXT GENERATION METRICS ---")
    print(f"BLEU-1: {metrics['bleu_1']:.4f}")
    print(f"BLEU-2: {metrics['bleu_2']:.4f}")
    print(f"BLEU-3: {metrics['bleu_3']:.4f}")
    print(f"BLEU-4: {metrics['bleu_4']:.4f}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    print(f"ROUGE: {metrics['rouge']:.4f}")
    print(f"CIDEr: {metrics['cider']:.4f}")
    
    print("="*80)

def print_sample_results(questions, predictions, ground_truths, images=None, num_samples=10):
    """Print sample results with detailed analysis"""
    
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    for i in range(min(num_samples, len(questions))):
        print(f"\nSample {i+1}:")
        if images and i < len(images):
            print(f"Image: {images[i]}")
        print(f"Question: {questions[i]}")
        print(f"Predicted: '{predictions[i]}'")
        print(f"Ground Truth: '{ground_truths[i]}'")
        
        # Detailed comparison
        pred_norm = predictions[i].strip().lower()
        gt_norm = ground_truths[i].strip().lower()
        
        is_exact_match = pred_norm == gt_norm
        is_partial_match = pred_norm in gt_norm or gt_norm in pred_norm
        is_empty = not pred_norm.strip()
        
        if is_exact_match:
            result = "✓ EXACT MATCH"
        elif is_partial_match:
            result = "~ PARTIAL MATCH"
        elif is_empty:
            result = "✗ EMPTY PREDICTION"
        else:
            result = "✗ NO MATCH"
            
        print(f"Result: {result}")
        print("-" * 60)
    
    print(f"\nShowing {min(num_samples, len(questions))} out of {len(questions)} total samples")
    print("="*80)

def analyze_answer_distribution(predictions, ground_truths):
    """Analyze the distribution of answers"""
    from collections import Counter
    
    print("\n" + "="*80)
    print("ANSWER DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Length analysis
    pred_lengths = [len(pred.split()) for pred in predictions if pred.strip()]
    truth_lengths = [len(truth.split()) for truth in ground_truths if truth.strip()]
    
    print(f"Average predicted answer length: {sum(pred_lengths)/len(pred_lengths):.2f} words" if pred_lengths else "No valid predictions")
    print(f"Average ground truth answer length: {sum(truth_lengths)/len(truth_lengths):.2f} words")
    
    if pred_lengths:
        print(f"Max predicted answer length: {max(pred_lengths)} words")
    print(f"Max ground truth answer length: {max(truth_lengths)} words")
    
    # Most common answers
    pred_counter = Counter([pred.strip().lower() for pred in predictions if pred.strip()])
    truth_counter = Counter([truth.strip().lower() for truth in ground_truths if truth.strip()])
    
    print("\nTop 10 most common predictions:")
    for answer, count in pred_counter.most_common(10):
        print(f"  '{answer}': {count} ({count/len(predictions)*100:.1f}%)")
    
    print("\nTop 10 most common ground truths:")
    for answer, count in truth_counter.most_common(10):
        print(f"  '{answer}': {count} ({count/len(ground_truths)*100:.1f}%)")
    
    print("="*80)