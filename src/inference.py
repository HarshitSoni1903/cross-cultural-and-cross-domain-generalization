"""
Inference module for loading trained models and making predictions.
Supports both classification (3-class sentiment) and regression (continuous ratings).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Union, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import XLMROBERTaRating


class SentimentPredictor:
    """Sentiment predictor for loading and using trained models."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize sentiment predictor.
        
        Args:
            model_path: Path to saved model directory
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detects if None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model (task_type will be auto-detected from saved config)
        self.model = XLMROBERTaRating.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Detect task type from model
        self.task_type = self.model.task_type
        print(f"Task type: {self.task_type}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Class names for classification
        self.class_names = ['Negative', 'Neutral', 'Positive']
        
        print("Model loaded successfully!")
    
    def predict_single(
        self,
        review_body: str,
        review_title: str = ""
    ) -> Dict:
        """
        Predict sentiment for a single review.
        
        Args:
            review_body: Review body text (required)
            review_title: Review title text (optional)
            
        Returns:
            Dictionary with:
                - For classification:
                    - prediction: Class index (0=Negative, 1=Neutral, 2=Positive)
                    - class_name: Class name (Negative/Neutral/Positive)
                    - probabilities: Probabilities for each class
                    - logits: Raw logits
                - For regression:
                    - prediction: Predicted star rating (1-5)
                    - raw_score: Raw unclipped regression score
                - text: Combined review text
        """
        # Combine title and body (if title provided)
        if review_title:
            review_text = f"{review_title}\n{review_body}"
        else:
            review_text = review_body
        
        # Tokenize
        encoding = self.tokenizer(
            review_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        result = {
            'text': review_text,
            'task_type': self.task_type
        }
        
        if self.task_type == 'classification':
            # Classification output
            prediction_idx = output['predictions'].item()
            logits = output['logits'].cpu()
            probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
            
            result.update({
                'prediction': int(prediction_idx),
                'class_name': self.class_names[prediction_idx],
                'probabilities': {
                    self.class_names[i]: prob 
                    for i, prob in enumerate(probabilities)
                },
                'logits': logits.squeeze().tolist()
            })
        else:
            # Regression output
            prediction = output['predictions'].item()
            raw_score = output['raw_score'].item()
            
            result.update({
                'prediction': float(prediction),
                'raw_score': float(raw_score)
            })
        
        return result
    
    def predict_batch(
        self,
        reviews: List[Dict]
    ) -> List[Dict]:
        """
        Predict sentiment for multiple reviews.
        
        Args:
            reviews: List of dicts with 'review_body' key (and optionally 'review_title')
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for review in reviews:
            result = self.predict_single(
                review_body=review['review_body'],
                review_title=review.get('review_title', '')
            )
            results.append(result)
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model directory'
    )
    parser.add_argument(
        '--body',
        type=str,
        required=True,
        help='Review body text (required)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default="",
        help='Review title text (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SentimentPredictor(args.model_path, device=args.device)
    
    # Single prediction
    result = predictor.predict_single(
        review_body=args.body,
        review_title=args.title
    )
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    
    if result['task_type'] == 'classification':
        print(f"Predicted Class: {result['class_name']} (Class {result['prediction']})")
        print(f"\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    else:
        print(f"Predicted Stars: {result['prediction']:.2f}/5")
        print(f"Raw Score: {result['raw_score']:.4f}")
    
    print("\nReview Text:")
    print("-"*50)
    print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
    print("="*50)


if __name__ == "__main__":
    main()

