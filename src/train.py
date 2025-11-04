"""
Training script for XLM-RoBERTa with regression head for rating prediction.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from data_preprocessing import create_dataloaders
from model import XLMROBERTaRating


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, config: Dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup model
        self.model = XLMROBERTaRating(
            model_name=config['model']['base_model'],
            num_classes=config['model']['num_labels']
        )
        self.model.to(self.device)
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
        
        # Setup data
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Create output directory with timestamp and language info
        base_output_dir = Path(config['data']['output_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        languages_str = "_".join(config['data']['train_languages'])
        self.output_dir = base_output_dir / f"{languages_str}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
    def setup_data(self):
        """Setup train, validation, and test dataloaders."""
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=".",
            languages=self.config['data']['train_languages'],
            tokenizer=self.tokenizer,
            max_length=self.config['training']['max_length'],
            batch_size=self.config['training']['batch_size']
        )
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Calculate total training steps
        num_training_steps = len(self.train_loader) * self.config['training']['num_epochs']
        
        # Setup learning rate scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = output['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / (step + 1)})
            
            # Logging
            if step % self.config['data']['logging_steps'] == 0:
                print(f"\nStep {step}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader, split_name: str = "Validation") -> Dict:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{split_name} Evaluation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = output['loss']
                
                # Collect predictions and labels
                predictions = output['predictions'].cpu()
                labels_cpu = labels.cpu()
                
                total_loss += loss.item()
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels_cpu.tolist())
        
        # Compute metrics
        all_predictions = torch.tensor(all_predictions, dtype=torch.float32)
        all_labels = torch.tensor(all_labels, dtype=torch.float32)
        
        # Convert to 1-5 range (model outputs 1-5, labels are 0-4)
        all_labels = all_labels + 1.0
        
        # Calculate regression metrics
        errors = all_predictions - all_labels
        
        # MAE (Mean Absolute Error)
        mae = torch.mean(torch.abs(errors)).item()
        
        # MSE (Mean Squared Error)
        mse = torch.mean(errors ** 2).item()
        
        # RMSE (Root Mean Squared Error)
        rmse = torch.sqrt(torch.mean(errors ** 2)).item()
        
        # R² (Coefficient of Determination)
        ss_res = torch.sum(errors ** 2)
        ss_tot = torch.sum((all_labels - torch.mean(all_labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2 = r2.item()
        
        # Accuracy (exact match for rounded predictions) - secondary metric
        predictions_rounded = torch.round(all_predictions)
        accuracy = (predictions_rounded == all_labels).float().mean().item()
        
        # Per-class MAE (useful for understanding performance across rating levels)
        class_mae = {}
        for class_label in range(1, 6):
            mask = all_labels == class_label
            if mask.sum() > 0:
                class_err = torch.abs(all_predictions[mask] - all_labels[mask])
                class_mae[class_label] = torch.mean(class_err).item()
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy,  # Rounded accuracy for reference
            'class_mae': class_mae
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict = None):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        print(f"\nSaved checkpoint to {checkpoint_dir}")
    
    def train(self):
        """Full training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        print(f"Languages: {self.config['data']['train_languages']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Number of epochs: {self.config['training']['num_epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print("="*50 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader, "Validation")
            print(f"\nValidation Metrics:")
            print(f"  Loss (MSE): {val_metrics['loss']:.4f}")
            print(f"  MAE: {val_metrics['mae']:.4f}")
            print(f"  RMSE: {val_metrics['rmse']:.4f}")
            print(f"  R²: {val_metrics['r2']:.4f}")
            print(f"  Accuracy (rounded): {val_metrics['accuracy']:.4f}")
            print(f"  Per-class MAE: {val_metrics['class_mae']}")
            
            # Save checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics)
                print(f"\nBest validation loss so far: {best_val_loss:.4f}")
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, "Test")
        print(f"\nTest Metrics:")
        print(f"  Loss (MSE): {test_metrics['loss']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  Accuracy (rounded): {test_metrics['accuracy']:.4f}")
        print(f"  Per-class MAE: {test_metrics['class_mae']}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train XLM-RoBERTa with CORAL')
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

