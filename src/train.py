"""
Training script for XLM-RoBERTa with rating prediction head.
Supports both regression (continuous ratings) and classification (3-class sentiment).
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from amazon_review_dataset import create_train_dataloader, create_amazon_review_dataloaders
from model import XLMROBERTaRating


def get_available_languages(data_path: str) -> List[str]:
    """
    Detect all available languages in the amazon_review data directory.
    
    Args:
        data_path: Base data path from config (e.g., "data").
                   Looks for {data_path}/amazon_review/language/train.jsonl
        
    Returns:
        List of available language codes that have train.jsonl files
    """
    # Path structure: {data_path}/amazon_review/language/train.jsonl
    amazon_review_path = Path(data_path) / "amazon_review"
    
    print(f"Looking for languages in: {amazon_review_path}")
    print(f"Path exists: {amazon_review_path.exists()}")
    
    available_languages = []
    
    if not amazon_review_path.exists():
        print(f"Error: Directory does not exist: {amazon_review_path}")
        return available_languages
    
    if not amazon_review_path.is_dir():
        print(f"Error: Path exists but is not a directory: {amazon_review_path}")
        return available_languages
    
    # Check each subdirectory for train.jsonl file
    try:
        items = list(amazon_review_path.iterdir())
        print(f"Found {len(items)} items in {amazon_review_path}")
        
        for lang_dir in items:
            if lang_dir.is_dir() and not lang_dir.name.startswith('.'):
                train_file = lang_dir / "train.jsonl"
                if train_file.exists():
                    available_languages.append(lang_dir.name)
                    print(f"  ✓ Found language: {lang_dir.name}")
    except Exception as e:
        print(f"Error scanning directory {amazon_review_path}: {e}")
    
    print(f"Total languages found: {len(available_languages)} - {available_languages}")
    return sorted(available_languages)


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, config: Dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Get task type and other configs
        self.task_type = config.get('model', {}).get('task_type', 'regression')  # 'classification' or 'regression'
        self.use_translation = config.get('data', {}).get('use_translation', False)
        num_classes = 3 if self.task_type == 'classification' else config['model'].get('num_labels', 5)
        
        print(f"Task type: {self.task_type}")
        print(f"Use translation: {self.use_translation}")
        
        # Setup model
        self.model = XLMROBERTaRating(
            model_name=config['model']['base_model'],
            num_classes=num_classes,
            task_type=self.task_type
        )
        self.model.to(self.device)
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
        
        # Initialize output directory variables (will be set after data setup)
        base_output_dir = Path(config['data']['output_dir'])
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup data
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
    
    def setup_data(self):
        """Setup train, validation, and test dataloaders using amazon_review_dataset."""
        # Get data_path from config
        data_path = self.config['data'].get('data_path', 'data')
        print(f"Using data path: {data_path}")
        
        # If use_translation is True, use all available languages' English translations
        # Otherwise, use only the specified languages with their original text
        if self.use_translation:
            # Get all available languages automatically
            print("Attempting to detect available languages...")
            available_languages = get_available_languages(data_path)
            
            # If no languages found, try to use train_languages as fallback
            if not available_languages:
                print("\n" + "="*60)
                print("WARNING: Could not automatically detect languages.")
                print("Attempting to use languages from config file as fallback...")
                print("="*60 + "\n")
                
                # Fallback to using train_languages from config
                fallback_languages = self.config['data'].get('train_languages', [])
                if fallback_languages:
                    print(f"Using fallback languages from config: {fallback_languages}")
                    languages_to_use = fallback_languages
                else:
                    raise ValueError(
                        f"No available languages found in {data_path}/amazon_review/ directory.\n"
                        f"Please ensure:\n"
                        f"  1. The directory exists at: {Path(data_path) / 'amazon_review'}\n"
                        f"  2. train.jsonl files exist for at least one language\n"
                        f"  3. Or set train_languages in config file\n"
                    )
            else:
                languages_to_use = available_languages
                print(f"Using translated data from ALL available languages: {languages_to_use}")
        else:
            # Use only specified languages with original text
            languages_to_use = self.config['data']['train_languages']
            print(f"Using original text from specified languages: {languages_to_use}")
        
        # Store languages used for later display
        self.languages_used = languages_to_use
        
        # Create train dataloader
        from amazon_review_dataset import create_amazon_review_dataloaders
        
        self.train_loader = create_amazon_review_dataloaders(
            data_dir=data_path,
            languages=languages_to_use,
            tokenizer=self.tokenizer,
            max_length=self.config['training']['max_length'],
            batch_size=self.config['training']['batch_size'],
            use_translation=self.use_translation,
            split='train'
        )
        
        # Create validation dataloader
        try:
            val_loader = create_amazon_review_dataloaders(
                data_dir=data_path,
                languages=languages_to_use,
                tokenizer=self.tokenizer,
                max_length=self.config['training']['max_length'],
                batch_size=self.config['training']['batch_size'],
                use_translation=self.use_translation,
                split='validation'
            )
            # Check if validation dataset is empty
            if len(val_loader.dataset) == 0:
                print("Warning: Validation dataset is empty. Using training data as validation data.")
                self.val_loader = self.train_loader
            else:
                self.val_loader = val_loader
                print(f"Validation samples: {len(self.val_loader.dataset)}")
        except Exception as e:
            print(f"Warning: Could not create validation dataloader: {e}")
            print("Using training data as validation data.")
            self.val_loader = self.train_loader
        
        # Create test dataloader
        try:
            test_loader = create_amazon_review_dataloaders(
                data_dir=data_path,
                languages=languages_to_use,
                tokenizer=self.tokenizer,
                max_length=self.config['training']['max_length'],
                batch_size=self.config['training']['batch_size'],
                use_translation=self.use_translation,
                split='test'
            )
            # Check if test dataset is empty
            if len(test_loader.dataset) == 0:
                print("Warning: Test dataset is empty. Using training data as test data.")
                self.test_loader = self.train_loader
            else:
                self.test_loader = test_loader
                print(f"Test samples: {len(self.test_loader.dataset)}")
        except Exception as e:
            print(f"Warning: Could not create test dataloader: {e}")
            print("Using training data as test data.")
            self.test_loader = self.train_loader
        
        # Now create output directory with correct language info
        if self.use_translation:
            available_languages = languages_to_use
            languages_str = "all" if len(available_languages) > 3 else "_".join(available_languages)
        else:
            languages_str = "_".join(languages_to_use)
        
        task_str = f"{self.task_type}_" if self.task_type != 'regression' else ""
        trans_str = "trans_" if self.use_translation else "orig_"
        self.output_dir = self.base_output_dir / f"{task_str}{trans_str}{languages_str}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
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
            
            # For regression, convert star_ratings (1-5) to labels (0-4)
            # For classification, use normalized labels (0-2)
            if self.task_type == 'regression':
                star_ratings = batch['star_ratings'].to(self.device)
                labels = star_ratings - 1  # Convert 1-5 to 0-4
            else:
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
        all_star_ratings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{split_name} Evaluation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # For regression, convert star_ratings (1-5) to labels (0-4)
                # For classification, use normalized labels (0-2)
                if self.task_type == 'regression':
                    star_ratings = batch['star_ratings'].to(self.device)
                    labels = star_ratings - 1  # Convert 1-5 to 0-4
                    all_star_ratings.extend(star_ratings.cpu().tolist())
                else:
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
        
        if self.task_type == 'classification':
            # Classification metrics
            all_predictions = torch.tensor(all_predictions, dtype=torch.long)
            all_labels = torch.tensor(all_labels, dtype=torch.long)
            
            # Convert to numpy for sklearn metrics
            pred_np = all_predictions.numpy()
            label_np = all_labels.numpy()
            
            # Overall accuracy
            accuracy = accuracy_score(label_np, pred_np)
            
            # Precision, Recall, F1 (macro and weighted)
            precision, recall, f1, support = precision_recall_fscore_support(
                label_np, pred_np, average=None, zero_division=0
            )
            precision_macro = precision_recall_fscore_support(
                label_np, pred_np, average='macro', zero_division=0
            )[0]
            recall_macro = precision_recall_fscore_support(
                label_np, pred_np, average='macro', zero_division=0
            )[1]
            f1_macro = precision_recall_fscore_support(
                label_np, pred_np, average='macro', zero_division=0
            )[2]
            f1_weighted = precision_recall_fscore_support(
                label_np, pred_np, average='weighted', zero_division=0
            )[2]
            
            # Confusion matrix
            cm = confusion_matrix(label_np, pred_np)
            
            # Per-class metrics
            class_metrics = {}
            class_names = ['Negative', 'Neutral', 'Positive']
            for i, class_name in enumerate(class_names):
                if i < len(precision):
                    class_metrics[class_name] = {
                        'precision': precision[i],
                        'recall': recall[i],
                        'f1': f1[i],
                        'support': support[i]
                    }
            
            metrics = {
                'loss': total_loss / len(dataloader),
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'class_metrics': class_metrics,
                'confusion_matrix': cm.tolist()
            }
        else:
            # Regression metrics - use original star ratings (1-5)
            all_predictions = torch.tensor(all_predictions, dtype=torch.float32)
            all_labels_stars = torch.tensor(all_star_ratings, dtype=torch.float32)
            
            # Calculate regression metrics
            errors = all_predictions - all_labels_stars
            
            # MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(errors)).item()
            
            # MSE (Mean Squared Error)
            mse = torch.mean(errors ** 2).item()
            
            # RMSE (Root Mean Squared Error)
            rmse = torch.sqrt(torch.mean(errors ** 2)).item()
            
            # R² (Coefficient of Determination)
            ss_res = torch.sum(errors ** 2)
            ss_tot = torch.sum((all_labels_stars - torch.mean(all_labels_stars)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r2 = r2.item()
            
            # Accuracy (exact match for rounded predictions)
            predictions_rounded = torch.round(all_predictions)
            accuracy = (predictions_rounded == all_labels_stars).float().mean().item()
            
            metrics = {
                'loss': total_loss / len(dataloader),
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'accuracy': accuracy
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
        print(f"Task type: {self.task_type}")
        print(f"Use translation: {self.use_translation}")
        
        # Show which languages are being used (from setup_data)
        if self.use_translation:
            print(f"Languages (ALL available, using translations): {self.languages_used}")
        else:
            print(f"Languages (specified, using original): {self.languages_used}")
        
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
            self._print_metrics(val_metrics)
            
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
        self._print_metrics(test_metrics)
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics based on task type."""
        if self.task_type == 'classification':
            print(f"  Loss (CE): {metrics['loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
            print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
            print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
            print(f"  Per-class metrics:")
            for class_name, class_metric in metrics['class_metrics'].items():
                print(f"    {class_name}: P={class_metric['precision']:.4f}, "
                      f"R={class_metric['recall']:.4f}, F1={class_metric['f1']:.4f}, "
                      f"Support={class_metric['support']}")
            print(f"  Confusion Matrix:")
            print(f"    {metrics['confusion_matrix']}")
        else:
            print(f"  Loss (MSE): {metrics['loss']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  Accuracy (rounded): {metrics['accuracy']:.4f}")


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

