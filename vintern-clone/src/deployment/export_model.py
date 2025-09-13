# src/deployment/export_model.py
import torch
from src.model.vintern_model import VinternModel

def export_model(checkpoint_path, output_path):
    """Export model for deployment"""
    
    # Load model
    model = VinternModel.from_pretrained(checkpoint_path)
    model.eval()
    
    # Convert to TorchScript (optional)
    example_input = {
        'images': torch.randn(1, 4, 3, 448, 448),
        'input_ids': torch.randint(0, 1000, (1, 100)),
        'attention_mask': torch.ones(1, 100)
    }
    
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(f"{output_path}/vintern_traced.pt")
    
    # Save normal PyTorch model
    torch.save(model.state_dict(), f"{output_path}/vintern_weights.pth")

if __name__ == "__main__":
    export_model("models/checkpoints/final_model", "models/deployed")