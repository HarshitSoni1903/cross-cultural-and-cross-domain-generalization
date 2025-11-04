"""
Inference module for loading trained models and making predictions.
"""

import argparse
from pathlib import Path
from typing import List, Union, Dict

import torch
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
        
        # Load model
        self.model = XLMROBERTaRating.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("Model loaded successfully!")
    
    def predict_single(
        self,
        review_title: str,
        review_body: str
    ) -> Dict:
        """
        Predict sentiment for a single review.
        
        Args:
            review_title: Review title text
            review_body: Review body text
            
        Returns:
            Dictionary with:
                - prediction: Predicted star rating (1-5)
                - text: Combined review text
                - raw_score: Raw unclipped regression score
        """
        # Combine title and body
        review_text = f"{review_title}\n{review_body}"
        
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
        
        prediction = output['predictions'].item()
        raw_score = output['raw_score'].item()
        
        result = {
            'prediction': prediction,
            'text': review_text,
            'raw_score': raw_score
        }
        
        return result
    
    def predict_batch(
        self,
        reviews: List[Dict]
    ) -> List[Dict]:
        """
        Predict sentiment for multiple reviews.
        
        Args:
            reviews: List of dicts with 'review_title' and 'review_body' keys
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for review in reviews:
            result = self.predict_single(
                review_title=review['review_title'],
                review_body=review['review_body']
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
        '--title',
        type=str,
        help='Review title'
    )
    parser.add_argument(
        '--body',
        type=str,
        help='Review body'
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
    
    if args.title and args.body:
        # Single prediction
        result = predictor.predict_single(
            review_title=args.title,
            review_body=args.body
        )
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Predicted Stars: {result['prediction']:.2f}/5")
        print(f"Raw Score: {result['raw_score']:.4f}")
        
        print("\nReview Text:")
        print("-"*50)
        print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
        print("="*50)
    else:
        # Example predictions
        print("\nRunning example predictions...")
        
        examples = [
            {
                'review_title': 'Great product!',
                'review_body': 'I absolutely love this product. It works perfectly and exceeded my expectations. Highly recommend!'
            },
            {
                'review_title': 'Not satisfied',
                'review_body': 'The product broke after just one week. Very disappointed with the quality. Would not recommend.'
            },
            {
                'review_title': 'Average',
                'review_body': 'It works okay, nothing special. The price is reasonable but I expected more features.'
            }
        ]
        
        results = predictor.predict_batch(examples)
        
        print("\n" + "="*50)
        print("EXAMPLE PREDICTIONS")
        print("="*50)
        
        for i, result in enumerate(results, 1):
            print(f"\nExample {i}:")
            print(f"Predicted Stars: {result['prediction']:.2f}/5")
            print(f"Raw Score: {result['raw_score']:.4f}")
            print(f"\nReview: {result['text'][:100]}...")


if __name__ == "__main__":
    main()

