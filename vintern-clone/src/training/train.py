# src/training/train.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoImageProcessor,
    TrainingArguments, Trainer
)
import wandb
from src.model.vintern_model import VinternModel
from src.data_processing.prepare_data import VinternDataset
from configs.training_config import TrainingConfig

def main():
    config = TrainingConfig()
    
    # Initialize wandb
    wandb.init(project="vintern-1b", name=config.run_name)
    
    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/InternViT-300M-448px")
    
    # Prepare datasets
    train_dataset = VinternDataset(
        config.train_data_path, 
        image_processor, 
        tokenizer, 
        config.max_length
    )
    
    val_dataset = VinternDataset(
        config.val_data_path,
        image_processor,
        tokenizer,
        config.max_length
    )
    
    # Initialize model
    model = VinternModel(config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True if config.mixed_precision == "fp16" else False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(f"{config.output_dir}/final_model")

if __name__ == "__main__":
    main()