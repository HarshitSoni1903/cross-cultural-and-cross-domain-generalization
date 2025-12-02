"""
Training script for XLM-RoBERTa with rating prediction head.
Supports both regression (continuous ratings) and classification (3-class sentiment).
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import json

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F

from amazon_review_dataset import create_train_dataloader, create_amazon_review_dataloaders
from model import XLMROBERTaRating, DualEncoderXLMROBERTaRating


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
    
    def __init__(self, config: Dict, config_path: str = None):
        """Initialize trainer with configuration."""
        self.config = config
        self.config_path = config_path  # Store path to original config file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Get task type and other configs
        self.task_type = config.get('model', {}).get('task_type', 'regression')  # 'classification' or 'regression'
        self.use_translation = config.get('data', {}).get('use_translation', False)
        self.training_schema = config.get('model', {}).get('training_schema', 'single')  # 'single' or 'dual_encoder'
        num_classes = 3 if self.task_type == 'classification' else config['model'].get('num_labels', 5)
        
        print(f"Task type: {self.task_type}")
        print(f"Use translation: {self.use_translation}")
        print(f"Training schema: {self.training_schema}")

        # Optional KL regularization between frozen (NLND) encoder and LD encoder
        training_cfg = config.get('training', {})
        self.use_ld_kl_penalty = training_cfg.get('use_ld_kl_penalty', False)
        self.ld_kl_weight = training_cfg.get('ld_kl_weight', 0.0)
        # Clip the KL contribution so it cannot dominate the main task loss
        # (acts as an upper bound on the raw KL value before weighting).
        self.ld_kl_clip = training_cfg.get('ld_kl_clip', 1.0)
        if self.training_schema != 'dual_encoder' and self.use_ld_kl_penalty:
            print("Warning: use_ld_kl_penalty is enabled but training_schema is not 'dual_encoder'. "
                  "The KL penalty will be ignored.")
        if self.use_ld_kl_penalty:
            print(f"LD KL penalty enabled with weight {self.ld_kl_weight}")
        
        # Setup model based on training schema
        if self.training_schema == 'dual_encoder':
            # Dual-encoder mode: requires pre-trained encoder path
            pretrained_path = config.get('model', {}).get('pretrained_encoder_path', '')
            if not pretrained_path:
                raise ValueError(
                    "pretrained_encoder_path must be provided in config when training_schema='dual_encoder'"
                )
            
            # Dual-encoder only supports classification
            if self.task_type != 'classification':
                raise ValueError("Dual-encoder mode only supports classification task now")
            
            # Check if baseline checkpoint should be used for encoder initialization
            use_baseline_checkpoint = config.get('model', {}).get('use_baseline_checkpoint_for_encoder', False)
            baseline_checkpoint_path = config.get('model', {}).get('baseline_checkpoint_path', '')
            
            if use_baseline_checkpoint:
                if not baseline_checkpoint_path:
                    raise ValueError(
                        "baseline_checkpoint_path must be provided in config when use_baseline_checkpoint_for_encoder=true"
                    )
                print(f"Baseline checkpoint enabled: will load new encoder from {baseline_checkpoint_path}")
            
            # Get classifier fusion method
            classifier_fusion_method = config.get('model', {}).get('classifier_fusion_method', 'concat')
            if classifier_fusion_method not in ['concat', 'residual']:
                raise ValueError(f"classifier_fusion_method must be 'concat' or 'residual', got '{classifier_fusion_method}'")

            # Optional NLND gating & dropout/masking controls for dual-encoder residual fusion.
            # - nlnd_drop_prob: branch-dropout probability on NLND path
            # - use_ld_masking: if true, apply gated masking at the NLND embedding level instead of only at logits
            nlnd_drop_prob = float(config.get('model', {}).get('nlnd_drop_prob', 0.0))
            use_ld_masking = bool(config.get('model', {}).get('use_ld_masking', False))
            if nlnd_drop_prob < 0.0 or nlnd_drop_prob >= 1.0:
                raise ValueError(f"nlnd_drop_prob must be in [0, 1), got {nlnd_drop_prob}")
            
            # Language embeddings option
            use_language_embeddings = bool(config.get('model', {}).get('use_language_embeddings', False))
            
            print(f"Initializing dual-encoder model...")
            print(f"Pre-trained encoder path: {pretrained_path}")
            print(f"Classifier fusion method: {classifier_fusion_method}")
            print(f"NLND drop prob: {nlnd_drop_prob}")
            print(f"Use LD masking on NLND embedding: {use_ld_masking}")
            print(f"Use language embeddings: {use_language_embeddings}")
            
            self.model = DualEncoderXLMROBERTaRating(
                pretrained_encoder_path=pretrained_path,
                base_model_name=config['model']['base_model'],
                num_classes=num_classes,
                freeze_pretrained=True,
                baseline_checkpoint_path=baseline_checkpoint_path if use_baseline_checkpoint else None,
                classifier_fusion_method=classifier_fusion_method,
                nlnd_drop_prob=nlnd_drop_prob,
                use_ld_masking=use_ld_masking,
                use_language_embeddings=use_language_embeddings,
            )
            self.model.to(self.device)
        else:
            # Single encoder mode (standard)
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
        
        # Setup TensorBoard logging
        # Create log directory in outputs/log with the same name as output_dir
        log_dir = self.base_output_dir / self.output_dir.name / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.log_dir = log_dir
        print(f"TensorBoard logs will be saved to: {log_dir}")
    
    def setup_data(self):
        """Setup train, validation, and test dataloaders using amazon_review_dataset."""
        # Get data_path from config
        data_path = self.config['data'].get('data_path', 'data')
        print(f"Using data path: {data_path}")

        # Select languages according to config priority
        if self.use_translation:
            config_languages = self.config['data'].get('train_languages', [])
            if config_languages:
                print(f"Using translated data from config-specified languages: {config_languages}")
                languages_to_use = config_languages
            else:
                print("No train_languages specified in config. Attempting to detect available languages...")
                available_languages = get_available_languages(data_path)
                if available_languages:
                    print(f"Using translated data from ALL available languages: {available_languages}")
                    languages_to_use = available_languages
                else:
                    raise ValueError(
                        f"No available languages found in {data_path}/amazon_review/ directory.\n"
                        f"Please ensure:\n"
                        f"  1. The directory exists at: {Path(data_path) / 'amazon_review'}\n"
                        f"  2. train.jsonl files exist for at least one language\n"
                        f"  3. Or set train_languages in config file\n"
                    )
        else:
            # Use only specified languages with original text
            languages_to_use = self.config['data'].get('train_languages', [])
            if not languages_to_use:
                raise ValueError(
                    "train_languages must be specified for non-translation (original text) mode."
                )
            print(f"Using original text from specified languages: {languages_to_use}")
        
        # Store languages used for later display
        self.languages_used = languages_to_use
        
        # Get domain_info from config
        domain_info = self.config.get('data', {}).get('domain_info', False)
        
        self.train_loader = create_amazon_review_dataloaders(
            data_dir=data_path,
            languages=languages_to_use,
            tokenizer=self.tokenizer,
            max_length=self.config['training']['max_length'],
            batch_size=self.config['training']['batch_size'],
            use_translation=self.use_translation,
            split='train',
            domain_info=domain_info,
            training_schema=self.training_schema
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
                split='validation',
                domain_info=domain_info,
                training_schema=self.training_schema
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
                split='test',
                domain_info=domain_info,
                training_schema=self.training_schema
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
        schema_str = "dual_" if self.training_schema == 'dual_encoder' else ""
        self.output_dir = self.base_output_dir / f"{schema_str}{task_str}{trans_str}{languages_str}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # For dual-encoder, only optimize new encoder and classifier (pre-trained encoder is frozen)
        if self.training_schema == 'dual_encoder':
            # Only optimize parameters that require gradients
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params_count = sum(p.numel() for p in trainable_params)
            frozen_params_count = total_params - trainable_params_count
            print(f"Trainable parameters: {trainable_params_count:,}")
            print(f"Frozen parameters: {frozen_params_count:,}")
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
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
    
    def train_epoch(self, epoch: int, global_step_start: int = 0, best_val_loss: float = float('inf')):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            global_step_start: Starting global step number (for step-based evaluation)
            best_val_loss: Current best validation loss (for checkpoint saving)
            
        Returns:
            Tuple of (avg_loss, updated best_val_loss)
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        eval_steps = self.config['data'].get('eval_steps', None)
        
        for step, batch in enumerate(progress_bar):
            # Get labels (always classification labels for dual-encoder, otherwise based on task_type)
            if self.training_schema == 'dual_encoder':
                # Dual-encoder always uses classification labels
                labels = batch['labels'].to(self.device)
            elif self.task_type == 'regression':
                star_ratings = batch['star_ratings'].to(self.device)
                labels = star_ratings - 1  # Convert 1-5 to 0-4
            else:
                labels = batch['labels'].to(self.device)
            
            # Forward pass based on training schema
            if self.training_schema == 'dual_encoder':
                # Dual-encoder: 
                # - Frozen encoder (pretrained): uses translated text (or original text for English reviews)
                # - Trainable encoder: uses original text
                # Note: For English reviews, both encoders use the same original English text
                input_ids_translated = batch['input_ids_translated'].to(self.device)
                attention_mask_translated = batch['attention_mask_translated'].to(self.device)
                input_ids_original = batch['input_ids_original'].to(self.device)
                attention_mask_original = batch['attention_mask_original'].to(self.device)
                
                # Get language codes from batch
                languages = batch.get('languages', None)
                
                output = self.model(
                    input_ids_translated=input_ids_translated,
                    attention_mask_translated=attention_mask_translated,
                    input_ids_original=input_ids_original,
                    attention_mask_original=attention_mask_original,
                    labels=labels,
                    languages=languages
                )
            else:
                # Single encoder: standard forward pass
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = output['loss']

            # Optional KL penalty between LD encoder (new_encoder) and frozen NLND encoder
            if (
                self.training_schema == 'dual_encoder'
                and self.use_ld_kl_penalty
                and self.ld_kl_weight > 0.0
            ):
                pretrained_pooled = output.get('pretrained_pooled')
                new_pooled = output.get('new_pooled')
                if pretrained_pooled is not None and new_pooled is not None:
                    # Treat pooled outputs as unnormalized logits over hidden dimension
                    # KL(new || pretrained) where pretrained acts as (frozen) target distribution
                    log_p_new = F.log_softmax(new_pooled, dim=-1)
                    p_pretrained = F.softmax(pretrained_pooled.detach(), dim=-1)
                    kl_div = F.kl_div(log_p_new, p_pretrained, reduction='batchmean')
                    # Clip KL so auxiliary term cannot overwhelm main task loss
                    kl_div_clipped = torch.clamp(kl_div, max=self.ld_kl_clip)
                    loss = loss - self.ld_kl_weight * kl_div_clipped
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / (step + 1)})
            
            # Calculate global step for TensorBoard
            global_step = global_step_start + step
            
            # TensorBoard logging
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], global_step)
            
            # Log residual loss components if available
            if self.training_schema == 'dual_encoder':
                if self.model.classifier_fusion_method == "residual":
                    loss_components = output.get('loss_components')
                    if loss_components is not None:
                        self.writer.add_scalar('Train/Residual/Loss_NLND', loss_components['loss_nlnd'].item(), global_step)
                        self.writer.add_scalar('Train/Residual/Loss_Combined', loss_components['loss_combined'].item(), global_step)
                        self.writer.add_scalar('Train/Residual/Penalty_Combined', loss_components['penalty_combined'].item(), global_step)
                        self.writer.add_scalar('Train/Residual/Reward', loss_components['reward'].item(), global_step)
                        self.writer.add_scalar('Train/Residual/Loss_Decrease', loss_components['loss_decrease'].item(), global_step)
                elif self.model.classifier_fusion_method == "concat":
                    loss_components = output.get('loss_components')
                    if loss_components is not None:
                        self.writer.add_scalar('Train/Concat/Loss', loss_components['loss'].item(), global_step)
                        self.writer.add_scalar('Train/Concat/Penalty', loss_components['penalty'].item(), global_step)
            # Evaluate at specified steps
            if eval_steps and global_step > 0 and global_step % eval_steps == 0:
                print(f"\n{'='*50}")
                print(f"Evaluation at step {global_step}")
                print(f"{'='*50}")
                val_metrics = self.evaluate(self.val_loader, "Validation", step=global_step)
                print(f"\nValidation Metrics:")
                self._print_metrics(val_metrics)
                
                # Save checkpoint if best validation loss
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(step=global_step, metrics=val_metrics)
                    print(f"\nBest validation loss so far: {best_val_loss:.4f}")
            
            # Logging
            if step % self.config['data']['logging_steps'] == 0:
                print(f"\nStep {global_step}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}")
        
        avg_loss = total_loss / len(self.train_loader)
        # Log average training loss for the epoch
        self.writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        return avg_loss, best_val_loss
    
    def evaluate(self, dataloader: DataLoader, split_name: str = "Validation", step: int = None, epoch: int = None) -> Dict:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_star_ratings = []
        
        # Track residual loss components if applicable
        total_loss_nlnd = 0.0
        total_loss_combined = 0.0
        total_penalty_combined = 0.0
        total_reward = 0.0
        total_loss_decrease = 0.0
        num_batches_with_components = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{split_name} Evaluation"):
                # Get labels
                if self.training_schema == 'dual_encoder':
                    # Dual-encoder always uses classification labels
                    labels = batch['labels'].to(self.device)
                elif self.task_type == 'regression':
                    star_ratings = batch['star_ratings'].to(self.device)
                    labels = star_ratings - 1  # Convert 1-5 to 0-4
                    all_star_ratings.extend(star_ratings.cpu().tolist())
                else:
                    labels = batch['labels'].to(self.device)
                
                # Forward pass based on training schema
                if self.training_schema == 'dual_encoder':
                    # Dual-encoder:
                    # - Frozen encoder (pretrained): uses translated text (or original text for English reviews)
                    # - Trainable encoder: uses original text
                    # Note: For English reviews, both encoders use the same original English text
                    input_ids_translated = batch['input_ids_translated'].to(self.device)
                    attention_mask_translated = batch['attention_mask_translated'].to(self.device)
                    input_ids_original = batch['input_ids_original'].to(self.device)
                    attention_mask_original = batch['attention_mask_original'].to(self.device)
                    
                    # Get language codes from batch
                    languages = batch.get('languages', None)
                    
                    output = self.model(
                        input_ids_translated=input_ids_translated,
                        attention_mask_translated=attention_mask_translated,
                        input_ids_original=input_ids_original,
                        attention_mask_original=attention_mask_original,
                        labels=labels,
                        languages=languages
                    )
                else:
                    # Single encoder: standard forward pass
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                loss = output['loss']
                
                # Collect residual loss components if available
                loss_components = output.get('loss_components')
                if loss_components is not None:
                    if self.model.classifier_fusion_method == "residual":
                        total_loss_nlnd += loss_components['loss_nlnd'].item()
                        total_loss_combined += loss_components['loss_combined'].item()
                        total_penalty_combined += loss_components['penalty_combined'].item()
                        total_reward += loss_components['reward'].item()
                        total_loss_decrease += loss_components['loss_decrease'].item()
                        num_batches_with_components += 1
                    elif self.model.classifier_fusion_method == "concat":
                        total_loss += loss_components['loss'].item()
                        total_penalty += loss_components['penalty'].item()
                        num_batches_with_components += 1

                # Apply the same KL penalty in evaluation loss if enabled
                if (
                    self.training_schema == 'dual_encoder'
                    and self.use_ld_kl_penalty
                    and self.ld_kl_weight > 0.0
                ):
                    pretrained_pooled = output.get('pretrained_pooled')
                    new_pooled = output.get('new_pooled')
                    if pretrained_pooled is not None and new_pooled is not None:
                        log_p_new = F.log_softmax(new_pooled, dim=-1)
                        p_pretrained = F.softmax(pretrained_pooled, dim=-1)
                        kl_div = F.kl_div(log_p_new, p_pretrained, reduction='batchmean')
                        kl_div_clipped = torch.clamp(kl_div, max=self.ld_kl_clip)
                        loss = loss + self.ld_kl_weight * kl_div_clipped
                
                # Collect predictions and labels
                predictions = output['predictions'].cpu()
                labels_cpu = labels.cpu()
                
                total_loss += loss.item()
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels_cpu.tolist())
        
        # Determine effective task type (dual-encoder is always classification)
        effective_task_type = 'classification' if self.training_schema == 'dual_encoder' else self.task_type
        
        if effective_task_type == 'classification':
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
            
            # TensorBoard logging for classification metrics
            # Use step if provided, otherwise fall back to epoch
            log_step = step if step is not None else epoch
            if log_step is not None:
                tag_prefix = f"{split_name}/"
                self.writer.add_scalar(f'{tag_prefix}Loss', metrics['loss'], log_step)
                self.writer.add_scalar(f'{tag_prefix}Accuracy', metrics['accuracy'], log_step)
                self.writer.add_scalar(f'{tag_prefix}Precision_Macro', metrics['precision_macro'], log_step)
                self.writer.add_scalar(f'{tag_prefix}Recall_Macro', metrics['recall_macro'], log_step)
                self.writer.add_scalar(f'{tag_prefix}F1_Macro', metrics['f1_macro'], log_step)
                self.writer.add_scalar(f'{tag_prefix}F1_Weighted', metrics['f1_weighted'], log_step)
                
                # Log residual loss components if available
                if num_batches_with_components > 0:
                    self.writer.add_scalar(f'{tag_prefix}Residual/Loss_NLND', total_loss_nlnd / num_batches_with_components, log_step)
                    self.writer.add_scalar(f'{tag_prefix}Residual/Loss_Combined', total_loss_combined / num_batches_with_components, log_step)
                    self.writer.add_scalar(f'{tag_prefix}Residual/Penalty_Combined', total_penalty_combined / num_batches_with_components, log_step)
                    self.writer.add_scalar(f'{tag_prefix}Residual/Reward', total_reward / num_batches_with_components, log_step)
                    self.writer.add_scalar(f'{tag_prefix}Residual/Loss_Decrease', total_loss_decrease / num_batches_with_components, log_step)
                
                # Per-class metrics
                for class_name, class_metric in class_metrics.items():
                    self.writer.add_scalar(f'{tag_prefix}Precision_{class_name}', class_metric['precision'], log_step)
                    self.writer.add_scalar(f'{tag_prefix}Recall_{class_name}', class_metric['recall'], log_step)
                    self.writer.add_scalar(f'{tag_prefix}F1_{class_name}', class_metric['f1'], log_step)
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
            
            # TensorBoard logging for regression metrics
            # Use step if provided, otherwise fall back to epoch
            log_step = step if step is not None else epoch
            if log_step is not None:
                tag_prefix = f"{split_name}/"
                self.writer.add_scalar(f'{tag_prefix}Loss', metrics['loss'], log_step)
                self.writer.add_scalar(f'{tag_prefix}MAE', metrics['mae'], log_step)
                self.writer.add_scalar(f'{tag_prefix}MSE', metrics['mse'], log_step)
                self.writer.add_scalar(f'{tag_prefix}RMSE', metrics['rmse'], log_step)
                self.writer.add_scalar(f'{tag_prefix}R2', metrics['r2'], log_step)
                self.writer.add_scalar(f'{tag_prefix}Accuracy', metrics['accuracy'], log_step)
        
        return metrics
    
    def save_checkpoint(self, step: int = None, epoch: int = None, metrics: Dict = None):
        """Save model checkpoint."""
        if step is not None:
            checkpoint_dir = self.output_dir / f"checkpoint-step-{step}"
        elif epoch is not None:
            checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        else:
            raise ValueError("Either step or epoch must be provided")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        print(f"\nSaved checkpoint to {checkpoint_dir}")
    
    def save_test_results(self, test_metrics: Dict):
        """Save test results to JSON file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # Add metadata to metrics
        results = {
            'training_schema': self.training_schema,
            'task_type': self.task_type,
            'languages': self.languages_used,
            'use_translation': self.use_translation,
            'test_samples': len(self.test_loader.dataset),
            'metrics': convert_to_serializable(test_metrics)
        }
        
        # Add pretrained encoder path and classifier fusion method for dual-encoder mode
        if self.training_schema == 'dual_encoder':
            results['pretrained_encoder_path'] = self.config.get('model', {}).get('pretrained_encoder_path', '')
            results['classifier_fusion_method'] = self.config.get('model', {}).get('classifier_fusion_method', 'concat')
        
        # Save to JSON file
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved to {results_path}")
    
    def save_config(self):
        """Save the config file used for training to the output directory."""
        config_path = self.output_dir / "train_config.yaml"
        
        # If original config file exists, copy it
        if self.config_path and Path(self.config_path).exists():
            import shutil
            shutil.copy2(self.config_path, config_path)
            print(f"\nConfig file copied to {config_path}")
        else:
            # Otherwise, save the config dict as YAML
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            print(f"\nConfig file saved to {config_path}")
    
    def train(self):
        """Full training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        print(f"Training schema: {self.training_schema}")
        print(f"Task type: {self.task_type}")
        print(f"Use translation: {self.use_translation}")
        
        if self.training_schema == 'dual_encoder':
            classifier_fusion_method = self.config.get('model', {}).get('classifier_fusion_method', 'concat')
            print(f"Dual-encoder mode:")
            print(f"  - Pre-trained encoder: frozen, processes translated text (original text for English)")
            print(f"  - New encoder: trainable, processes original text")
            if classifier_fusion_method == "concat":
                print(f"  - Classifier: concatenates features from both encoders")
            else:  # residual
                print(f"  - Classifier: NLND classifier + LD residual classifier (additive logits)")
            print(f"  - Note: For English reviews, both encoders use the same original English text")
        
        # Show which languages are being used (from setup_data)
        if self.use_translation:
            print(f"Languages (ALL available, using translations): {self.languages_used}")
        else:
            print(f"Languages (specified, using original): {self.languages_used}")
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Number of epochs: {self.config['training']['num_epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        
        # Get evaluation configuration
        eval_steps = self.config['data'].get('eval_steps', None)
        if eval_steps:
            print(f"Evaluation frequency: every {eval_steps} steps")
        else:
            print("Evaluation frequency: at the end of each epoch")
        print("="*50 + "\n")
        
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            # Train epoch (evaluation happens inside train_epoch if eval_steps is set)
            train_loss, best_val_loss = self.train_epoch(epoch, global_step_start=global_step, best_val_loss=best_val_loss)
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Update global step
            global_step += len(self.train_loader)
            
            # Evaluate at end of epoch only if eval_steps is not set (fallback to epoch-based evaluation)
            if not eval_steps:
                val_metrics = self.evaluate(self.val_loader, "Validation", epoch=epoch)
                print(f"\nValidation Metrics:")
                self._print_metrics(val_metrics)
                
                # Save checkpoint
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch=epoch, metrics=val_metrics)
                    print(f"\nBest validation loss so far: {best_val_loss:.4f}")
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        
        # Final evaluation on test set (use global step for logging)
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, "Test", step=global_step)
        print(f"\nTest Metrics:")
        self._print_metrics(test_metrics)
        
        # Save test results to file
        self.save_test_results(test_metrics)
        
        # Save config file used for training
        self.save_config()
        
        # Close TensorBoard writer
        self.writer.close()
        print(f"\nTensorBoard logs saved to: {self.log_dir}")
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics based on task type."""
        # Determine effective task type (dual-encoder is always classification)
        effective_task_type = 'classification' if self.training_schema == 'dual_encoder' else self.task_type
        
        if effective_task_type == 'classification':
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
    
    # Create trainer and train (pass config path for saving)
    trainer = Trainer(config, config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()

