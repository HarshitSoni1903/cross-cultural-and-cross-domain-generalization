"""
Model architecture with XLM-RoBERTa base and rating prediction head.
Supports both regression (continuous ratings) and classification (3-class: Negative, Neutral, Positive).
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoConfig


class RatingClassificationHead(nn.Module):
    """
    Classification head for predicting sentiment categories.
    
    Outputs logits for 3 classes: Negative (0), Neutral (1), Positive (2).
    """
    
    def __init__(self, input_dim: int, num_classes: int = 3):
        """
        Initialize classification head.
        
        Args:
            input_dim: Dimension of input features (e.g., 768 for XLM-RoBERTa)
            num_classes: Number of classes (default 3: Negative, Neutral, Positive)
        """
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, features: Tensor, labels: Tensor = None) -> dict:
        """
        Forward pass through classification head.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            labels: True labels for computing classification loss (optional)
            
        Returns:
            Dictionary containing:
                - logits: Raw logits [batch_size, num_classes]
                - predictions: Predicted class indices [batch_size]
                - loss: Cross-entropy loss if labels provided
        """
        # Get logits
        logits = self.classifier(features)  # [batch_size, num_classes]
        
        # Get predicted class indices
        predictions = torch.argmax(logits, dim=-1)  # [batch_size]
        
        output = {
            'logits': logits,
            'predictions': predictions
        }
        
        # Compute Cross-Entropy loss if labels provided
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            output['loss'] = loss
        
        return output


class RatingRegressionHead(nn.Module):
    """
    Regression head for predicting continuous rating scores.
    
    Outputs float values in [0, 6] range which are clipped to [1, 5] for star ratings.
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize regression head.
        
        Args:
            input_dim: Dimension of input features (e.g., 768 for XLM-RoBERTa)
        """
        super().__init__()
        # Single output regression head
        self.regressor = nn.Linear(input_dim, 1)
        
    def forward(self, features: Tensor, labels: Tensor = None) -> dict:
        """
        Forward pass through regression head.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            labels: True labels for computing regression loss (optional)
            
        Returns:
            Dictionary containing:
                - raw_score: Raw regression output [batch_size]
                - predictions: Predicted ratings [batch_size] in range [1, 5] (clipped)
                - loss: MSE loss if labels provided
        """
        # Get raw regression score (unbounded)
        raw_score = self.regressor(features).squeeze(-1)  # [batch_size]
        
        # Transform to [0, 6] range using sigmoid and scale
        # sigmoid gives [0, 1], multiply by 6 gives [0, 6]
        score_0_6 = torch.sigmoid(raw_score) * 6.0
        
        # Clip to [1, 5] range for star ratings
        predictions = torch.clamp(score_0_6, 1.0, 5.0)
        
        output = {
            'raw_score': raw_score,
            'predictions': predictions
        }
        
        # Compute MSE loss if labels provided
        if labels is not None:
            loss = self.regression_loss(score_0_6, labels)
            output['loss'] = loss
        
        return output
    
    def regression_loss(self, predictions: Tensor, labels: Tensor) -> Tensor:
        """
        Compute MSE loss for regression.
        
        Args:
            predictions: Predicted scores [batch_size] in range [0, 6]
            labels: True labels in range [0, 4] (0-indexed, representing stars 1-5)
            
        Returns:
            MSE loss scalar
        """
        # Convert labels from [0, 4] to [1, 5] range
        labels_1_5 = labels.float() + 1.0
        
        # MSE loss between predictions and labels
        criterion = nn.MSELoss()
        loss = criterion(predictions, labels_1_5)
        
        return loss


class XLMROBERTaRating(nn.Module):
    """
    XLM-RoBERTa model with rating prediction head.
    Supports both regression (continuous ratings) and classification (3-class sentiment).
    """
    
    def __init__(
        self, 
        model_name: str = "FacebookAI/xlm-roberta-base", 
        num_classes: int = 5,
        task_type: str = "regression"
    ):
        """
        Initialize XLM-RoBERTa with rating prediction head.
        
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of rating classes (for classification, default 3; for regression, ignored)
            task_type: Type of task - "regression" or "classification"
        """
        super().__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes if task_type == "classification" else 5
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze embeddings if desired (optional, commented out)
        # for param in self.bert.embeddings.parameters():
        #     param.requires_grad = False
        
        # Choose head based on task type
        if task_type == "classification":
            self.rating_head = RatingClassificationHead(
                input_dim=self.config.hidden_size,
                num_classes=self.num_classes
            )
        else:  # regression
            self.rating_head = RatingRegressionHead(input_dim=self.config.hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the head layer."""
        if self.task_type == "classification":
            nn.init.xavier_uniform_(self.rating_head.classifier.weight)
            nn.init.zeros_(self.rating_head.classifier.bias)
        else:
            nn.init.xavier_uniform_(self.rating_head.regressor.weight)
            nn.init.zeros_(self.rating_head.regressor.bias)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor = None
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: True labels for computing loss (optional)
            
        Returns:
            Dictionary containing raw_score, predictions, and optionally loss
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Pass through regression head
        rating_output = self.rating_head(pooled_output, labels)
        
        return rating_output
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer to directory."""
        self.bert.save_pretrained(save_directory)
        head_filename = "classification_head.pt" if self.task_type == "classification" else "rating_head.pt"
        torch.save(self.rating_head.state_dict(), f"{save_directory}/{head_filename}")
        # Save task type for loading
        import json
        with open(f"{save_directory}/model_config.json", 'w') as f:
            json.dump({
                'task_type': self.task_type,
                'num_classes': self.num_classes
            }, f)
    
    @classmethod
    def from_pretrained(cls, save_directory: str, num_classes: int = None, task_type: str = None):
        """
        Load model from directory.
        
        Args:
            save_directory: Path to saved model directory
            num_classes: Number of classes (if None, loaded from config)
            task_type: Task type (if None, loaded from config)
            
        Returns:
            Loaded model instance
        """
        # Try to load config
        import json
        import os
        config_path = f"{save_directory}/model_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                task_type = task_type or config.get('task_type', 'regression')
                num_classes = num_classes or config.get('num_classes', 3 if task_type == 'classification' else 5)
        else:
            # Default to regression for backward compatibility
            task_type = task_type or 'regression'
            num_classes = num_classes or (3 if task_type == 'classification' else 5)
        
        model = cls(model_name=save_directory, num_classes=num_classes, task_type=task_type)
        head_filename = "classification_head.pt" if task_type == "classification" else "rating_head.pt"
        rating_path = f"{save_directory}/{head_filename}"
        if os.path.exists(rating_path):
            if torch.cuda.is_available():
                model.rating_head.load_state_dict(torch.load(rating_path))
            else:
                model.rating_head.load_state_dict(torch.load(rating_path, map_location='cpu'))
        return model


if __name__ == "__main__":
    # Test regression model
    print("Testing Regression Model:")
    model_reg = XLMROBERTaRating(num_classes=5, task_type='regression')
    
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 5, (batch_size,))  # 0-4 labels for regression
    
    output = model_reg(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"Output keys: {output.keys()}")
    print(f"Raw score shape: {output['raw_score'].shape}")
    print(f"Predictions: {output['predictions']}")
    print(f"Loss: {output['loss']:.4f}")
    
    # Test classification model
    print("\nTesting Classification Model:")
    model_cls = XLMROBERTaRating(num_classes=3, task_type='classification')
    
    labels_cls = torch.randint(0, 3, (batch_size,))  # 0-2 labels for classification
    
    output_cls = model_cls(input_ids=input_ids, attention_mask=attention_mask, labels=labels_cls)
    print(f"Output keys: {output_cls.keys()}")
    print(f"Logits shape: {output_cls['logits'].shape}")
    print(f"Predictions: {output_cls['predictions']}")
    print(f"Loss: {output_cls['loss']:.4f}")
    
    # Test without labels
    output_no_labels = model_cls(input_ids=input_ids, attention_mask=attention_mask)
    print(f"\nPredictions without labels: {output_no_labels['predictions']}")

