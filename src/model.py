"""
Model architecture with XLM-RoBERTa base and rating prediction head.
Supports both regression (continuous ratings) and classification (3-class: Negative, Neutral, Positive).
Also supports dual-encoder architecture with frozen pre-trained encoder and trainable encoder.
"""

import json
import os

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


class DualEncoderXLMROBERTaRating(nn.Module):
    """
    Dual-encoder XLM-RoBERTa model for cross-lingual sentiment analysis.
    
    Architecture:
    - Encoder 1 (frozen): Pre-trained encoder that processes translated text (review_body_en)
    - Encoder 2 (trainable): New encoder that processes original text (review_body)
    - Classifier: Two fusion methods supported:
        1. "concat": Concatenate embeddings from both encoders and use a single classifier
        2. "residual": Use NLND classifier on pretrained encoder output, LD classifier on new encoder output, add logits
    """
    
    def __init__(
        self,
        pretrained_encoder_path: str,
        base_model_name: str = "FacebookAI/xlm-roberta-base",
        num_classes: int = 3,
        freeze_pretrained: bool = True,
        baseline_checkpoint_path: str = None,
        classifier_fusion_method: str = "concat",
        nlnd_drop_prob: float = 0.0,
        use_ld_masking: bool = False,
    ):
        """
        Initialize dual-encoder model.
        
        Args:
            pretrained_encoder_path: Path to pre-trained model checkpoint (frozen encoder)
            base_model_name: HuggingFace model identifier for new encoder (used if baseline_checkpoint_path is None)
            num_classes: Number of classes (default 3 for classification)
            freeze_pretrained: Whether to freeze the pre-trained encoder (default True)
            baseline_checkpoint_path: Optional path to baseline checkpoint to load encoder from. If provided, the new encoder will be loaded from this checkpoint instead of initializing from scratch.
            classifier_fusion_method: "concat" or "residual". "concat" concatenates embeddings and uses one classifier. "residual" uses NLND classifier + LD residual classifier.
        """
        super().__init__()
        
        self.task_type = "classification"  # Dual-encoder only supports classification
        self.num_classes = num_classes
        self.pretrained_encoder_path = pretrained_encoder_path
        self.classifier_fusion_method = classifier_fusion_method
        # NLND gating & (optional) branch-dropout / masking
        # Initialized to -1 => sigmoid(-1) ~ 0.27, so we start leaned towards LD.
        self.nlnd_gate = nn.Parameter(torch.tensor(-1.0))
        self.nlnd_drop_prob = float(nlnd_drop_prob)
        self.use_ld_masking = bool(use_ld_masking)
        
        if classifier_fusion_method not in ["concat", "residual"]:
            raise ValueError(f"classifier_fusion_method must be 'concat' or 'residual', got '{classifier_fusion_method}'")
        
        # Load pre-trained encoder (frozen, inference only)
        print(f"Loading pre-trained encoder from {pretrained_encoder_path}")
        pretrained_model = XLMROBERTaRating.from_pretrained(pretrained_encoder_path)
        self.pretrained_encoder = pretrained_model.bert
        
        # Freeze pre-trained encoder
        if freeze_pretrained:
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            print("Pre-trained encoder frozen (inference only)")
        else:
            print("Pre-trained encoder trainable")
        
        # Create new trainable encoder
        if baseline_checkpoint_path:
            print(f"Loading new encoder from baseline checkpoint: {baseline_checkpoint_path}")
            baseline_model = XLMROBERTaRating.from_pretrained(baseline_checkpoint_path)
            self.new_encoder = baseline_model.bert
            # Use config from baseline encoder (same as baseline_model.config since it's the BERT config)
            self.config = baseline_model.config
            print("New encoder loaded from baseline checkpoint")
        else:
            print(f"Initializing new encoder from {base_model_name}")
            self.config = AutoConfig.from_pretrained(base_model_name)
            self.new_encoder = AutoModel.from_pretrained(base_model_name)
        
        # Setup classifier based on fusion method
        if classifier_fusion_method == "concat":
            # Classification head on concatenated features
            # Each encoder outputs hidden_size dim, so concatenated is 2 * hidden_size
            combined_dim = self.config.hidden_size * 2
            self.classifier = nn.Linear(combined_dim, num_classes)
            self.pretrained_classifier = None
            self.ld_classifier = None
        else:  # residual
            # Load NLND classifier from pretrained model
            print(f"Loading NLND classifier from pretrained model")
            self.pretrained_classifier = pretrained_model.rating_head.classifier
            # Freeze NLND classifier
            if freeze_pretrained:
                for param in self.pretrained_classifier.parameters():
                    param.requires_grad = False
                print("NLND classifier frozen")
            else:
                print("NLND classifier trainable")
            
            # Create new LD classifier for residual prediction
            print(f"Creating LD residual classifier")
            self.ld_classifier = nn.Linear(self.config.hidden_size, num_classes)
            self.classifier = None
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the classifier layer(s)."""
        if self.classifier_fusion_method == "concat":
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        else:  # residual
            # Initialize LD classifier weights
            nn.init.xavier_uniform_(self.ld_classifier.weight)
            nn.init.zeros_(self.ld_classifier.bias)
    
    def forward(
        self,
        input_ids_translated: Tensor,
        attention_mask_translated: Tensor,
        input_ids_original: Tensor,
        attention_mask_original: Tensor,
        labels: Tensor = None
    ) -> dict:
        """
        Forward pass through dual-encoder model.
        
        Args:
            input_ids_translated: Token IDs for translated text [batch_size, seq_len]
            attention_mask_translated: Attention mask for translated text [batch_size, seq_len]
            input_ids_original: Token IDs for original text [batch_size, seq_len]
            attention_mask_original: Attention mask for original text [batch_size, seq_len]
            labels: True labels for computing loss (optional)
            
        Returns:
            Dictionary containing logits, predictions, and optionally loss
        """
        # Forward through pre-trained encoder (frozen, translated text)
        # Set to eval mode and disable gradients for efficiency
        self.pretrained_encoder.eval()
        # We don't use no_grad here because we want to allow gradients through concatenation
        # but the encoder parameters are frozen so no gradients will flow to them anyway
        pretrained_outputs = self.pretrained_encoder(
            input_ids=input_ids_translated,
            attention_mask=attention_mask_translated,
            return_dict=True
        )
        pretrained_pooled = pretrained_outputs.last_hidden_state[:, 0, :].detach()  # [batch_size, hidden_size]
        
        # Forward through new encoder (trainable, original text)
        new_outputs = self.new_encoder(
            input_ids=input_ids_original,
            attention_mask=attention_mask_original,
            return_dict=True
        )
        new_pooled = new_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]


        logits_pretrained = None
        logits_new = None
    
        if self.classifier_fusion_method == "concat":
            # Split concat classifier weights: [W_pre | W_new]
            W = self.classifier.weight      # [num_classes, 2*hidden]
            b = self.classifier.bias        # [num_classes]
    
            hidden = self.config.hidden_size
            W_pre = W[:, :hidden]
            W_new = W[:, hidden:]
    
            logits_pretrained = pretrained_pooled @ W_pre.T + b
            logits_new        = new_pooled        @ W_new.T + b
    
        else:   # residual mode
            logits_pretrained = self.pretrained_classifier(pretrained_pooled)
            logits_new        = self.ld_classifier(new_pooled)
        
        # Compute logits based on fusion method
        if self.classifier_fusion_method == "concat":
            # Gate in (0, 1) controlling how much NLND contributes.
            gate = torch.sigmoid(self.nlnd_gate)
            
            # Optional gated masking applied at the NLND embedding level.
            if self.use_ld_masking:
                # Start by scaling with the global gate.
                mask_factor = gate
                # Optional branch-dropout (per example) on NLND path.
                if self.training and self.nlnd_drop_prob > 0.0:
                    m = torch.bernoulli(
                        torch.full(
                            (pretrained_pooled.size(0), 1),
                            1.0 - self.nlnd_drop_prob,
                            device=pretrained_pooled.device,
                        )
                    )
                    # Inverted dropout scaling to keep expectation fixed.
                    mask_factor = mask_factor * (m / (1.0 - self.nlnd_drop_prob))
                pretrained_pooled_for_concat = pretrained_pooled * mask_factor
            else:
                pretrained_pooled_for_concat = pretrained_pooled
            
            # Concatenate features from both encoders
            combined_features = torch.cat([pretrained_pooled_for_concat, new_pooled], dim=-1)  # [batch_size, 2*hidden_size]
            
            # Pass through classifier
            logits = self.classifier(combined_features)  # [batch_size, num_classes]
        elif self.classifier_fusion_method == "residual":  # residual
            # Gate in (0, 1) controlling how much NLND contributes.
            gate = torch.sigmoid(self.nlnd_gate)

            # Optional gated masking applied at the NLND embedding level.
            if self.use_ld_masking:
                # Start by scaling with the global gate.
                mask_factor = gate
                # Optional branch-dropout (per example) on NLND path.
                if self.training and self.nlnd_drop_prob > 0.0:
                    m = torch.bernoulli(
                        torch.full(
                            (pretrained_pooled.size(0), 1),
                            1.0 - self.nlnd_drop_prob,
                            device=pretrained_pooled.device,
                        )
                    )
                    # Inverted dropout scaling to keep expectation fixed.
                    mask_factor = mask_factor * (m / (1.0 - self.nlnd_drop_prob))
                pretrained_pooled_for_cls = pretrained_pooled * mask_factor
                logits_nlnd = self.pretrained_classifier(pretrained_pooled_for_cls)  # [batch_size, num_classes]
            else:
                # Standard NLND logits from frozen classifier.
                logits_nlnd = self.pretrained_classifier(pretrained_pooled)  # [batch_size, num_classes]
                # Optional branch-dropout directly on NLND logits.
                if self.training and self.nlnd_drop_prob > 0.0:
                    m = torch.bernoulli(
                        torch.full(
                            (logits_nlnd.size(0), 1),
                            1.0 - self.nlnd_drop_prob,
                            device=logits_nlnd.device,
                        )
                    )
                    logits_nlnd = logits_nlnd * m / (1.0 - self.nlnd_drop_prob)
                # Always apply global gate at logit level in this branch.
                logits_nlnd = gate * logits_nlnd

            # LD classifier on new encoder output (predicts residual)
            logits_residual = self.ld_classifier(new_pooled)  # [batch_size, num_classes]

            # Add logits together
            logits = logits_nlnd + logits_residual  # [batch_size, num_classes]
        else:
            raise ValueError(f"Invalid classifier fusion method: {self.classifier_fusion_method}")
            
        predictions = torch.argmax(logits, dim=-1)  # [batch_size]
        
        # Base outputs
        output = {
            'logits': logits,
            'predictions': predictions,
            # Expose pooled encoder outputs for optional auxiliary losses (e.g., KL regularization)
            'pretrained_pooled': pretrained_pooled,
            'new_pooled': new_pooled,
            'logits_pretrained': logits_pretrained,
            'logits_new': logits_new,     
        }
        
        # Compute Cross-Entropy loss if labels provided
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            output['loss'] = loss
        
        return output
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save new encoder
        self.new_encoder.save_pretrained(f"{save_directory}/new_encoder")
        
        # Save classifier(s) based on fusion method
        if self.classifier_fusion_method == "concat":
            torch.save(self.classifier.state_dict(), f"{save_directory}/classifier.pt")
        else:  # residual
            torch.save(self.ld_classifier.state_dict(), f"{save_directory}/ld_classifier.pt")
            # Note: pretrained_classifier is not saved as it comes from pretrained_encoder_path
        
        # Save config
        with open(f"{save_directory}/model_config.json", 'w') as f:
            json.dump({
                'task_type': self.task_type,
                'num_classes': self.num_classes,
                'model_type': 'dual_encoder',
                'pretrained_encoder_path': self.pretrained_encoder_path,
                'classifier_fusion_method': self.classifier_fusion_method
            }, f)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, save_directory: str, pretrained_encoder_path: str = None):
        """
        Load dual-encoder model from directory.
        
        Args:
            save_directory: Path to saved model directory
            pretrained_encoder_path: Path to pre-trained encoder (if not in config)
            
        Returns:
            Loaded model instance
        """
        # Load config
        config_path = f"{save_directory}/model_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                num_classes = config.get('num_classes', 3)
                pretrained_path = pretrained_encoder_path or config.get('pretrained_encoder_path')
                classifier_fusion_method = config.get('classifier_fusion_method', 'concat')
        else:
            num_classes = 3
            pretrained_path = pretrained_encoder_path
            classifier_fusion_method = 'concat'
        
        if pretrained_path is None:
            raise ValueError("pretrained_encoder_path must be provided")
        
        # Create model
        model = cls(
            pretrained_encoder_path=pretrained_path,
            num_classes=num_classes,
            freeze_pretrained=True,
            classifier_fusion_method=classifier_fusion_method
        )
        
        # Load new encoder
        new_encoder_path = f"{save_directory}/new_encoder"
        if os.path.exists(new_encoder_path):
            model.new_encoder = AutoModel.from_pretrained(new_encoder_path)
        
        # Load classifier(s) based on fusion method
        if classifier_fusion_method == "concat":
            classifier_path = f"{save_directory}/classifier.pt"
            if os.path.exists(classifier_path):
                if torch.cuda.is_available():
                    model.classifier.load_state_dict(torch.load(classifier_path))
                else:
                    model.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        else:  # residual
            ld_classifier_path = f"{save_directory}/ld_classifier.pt"
            if os.path.exists(ld_classifier_path):
                if torch.cuda.is_available():
                    model.ld_classifier.load_state_dict(torch.load(ld_classifier_path))
                else:
                    model.ld_classifier.load_state_dict(torch.load(ld_classifier_path, map_location='cpu'))
        
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

