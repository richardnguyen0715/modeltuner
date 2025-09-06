import torch
from bartphobeit.config import get_model_combinations
from bartphobeit.model import UniversalVietnameseVQAModel

def test_model_combination(combo_name, combo_config):
    """Test a specific model combination"""
    print(f"\n{'='*60}")
    print(f"Testing {combo_name}: {combo_config['description']}")
    print(f"{'='*60}")
    
    try:
        config = {
            'vision_model': combo_config['vision_model'],
            'text_model': combo_config['text_model'],
            'decoder_model': combo_config['decoder_model'],
            'hidden_dim': 768,
            'dropout_rate': 0.1,
            'pool_target_length': 24,
            'label_smoothing': 0.1,
            'device': 'cpu'  # Use CPU for testing
        }
        
        model = UniversalVietnameseVQAModel(config)
        
        # Test forward pass
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        question_input_ids = torch.randint(0, 1000, (batch_size, 32))
        question_attention_mask = torch.ones(batch_size, 32)
        answer_input_ids = torch.randint(0, 1000, (batch_size, 16))
        answer_attention_mask = torch.ones(batch_size, 16)
        
        # Training forward pass
        print("Testing training mode...")
        outputs = model(
            pixel_values=pixel_values,
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            answer_input_ids=answer_input_ids,
            answer_attention_mask=answer_attention_mask
        )
        print(f"✓ Training forward pass successful, loss: {outputs.loss.item():.4f}")
        
        # Inference forward pass
        print("Testing inference mode...")
        generated_ids = model(
            pixel_values=pixel_values,
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask
        )
        print(f"✓ Inference forward pass successful, output shape: {generated_ids.shape}")
        
        print(f"✅ {combo_name} combination works perfectly!")
        
    except Exception as e:
        print(f"❌ {combo_name} combination failed: {e}")

def main():
    """Test all model combinations"""
    print("🧪 Testing Universal Vietnamese VQA Model with all combinations")
    
    combinations = get_model_combinations()
    
    for combo_name, combo_config in combinations.items():
        test_model_combination(combo_name, combo_config)
    
    print(f"\n{'='*60}")
    print("🎉 Universal model testing completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()