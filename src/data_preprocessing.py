"""
Data preprocessing module for Amazon multilingual reviews dataset.
Handles loading, unzipping, and tokenizing the data for training.
"""

import gzip
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class AmazonReviewsDataset(Dataset):
    """Dataset class for Amazon multilingual reviews."""
    
    def __init__(
        self,
        data_dir: str,
        languages: List[str],
        split: str = "train",
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing amazon_reviews_multi folder
            languages: List of language codes to load (e.g., ['en', 'fr'])
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length for tokenization
        """
        self.data_dir = Path(data_dir)
        self.languages = languages
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data from all specified languages
        self._load_data()
    
    def _load_data(self):
        """Load and parse JSONL files from all specified languages."""
        for lang in self.languages:
            lang_dir = self.data_dir / f"amazon_reviews_multi" / lang
            jsonl_file = lang_dir / f"{self.split}.jsonl.gz"
            
            if not jsonl_file.exists():
                raise FileNotFoundError(f"Data file not found: {jsonl_file}")
            
            # Unzip and load data
            with gzip.open(jsonl_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # Combine review_title and review_body
                    review_text = f"{item['review_title']}\n{item['review_body']}"
                    self.data.append({
                        'review_id': item['review_id'],
                        'product_id': item['product_id'],
                        'text': review_text,
                        'stars': int(item['stars']),  # 1-5
                        'language': item['language']
                    })
        
        print(f"Loaded {len(self.data)} examples from {self.languages} ({self.split})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single data sample."""
        item = self.data[idx]
        
        # Prepare output
        output = {
            'review_id': item['review_id'],
            'product_id': item['product_id'],
            'text': item['text'],
            'stars': item['stars'],
            'labels': item['stars'] - 1  # Convert to 0-4 for indexing
        }
        
        # Tokenize if tokenizer is provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                item['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            output['input_ids'] = encoding['input_ids'].squeeze()
            output['attention_mask'] = encoding['attention_mask'].squeeze()
        
        return output


def create_dataloaders(
    data_dir: str,
    languages: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing amazon_reviews_multi folder
        languages: List of language codes (e.g., ['en', 'fr'])
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = AmazonReviewsDataset(
        data_dir=data_dir,
        languages=languages,
        split='train',
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = AmazonReviewsDataset(
        data_dir=data_dir,
        languages=languages,
        split='validation',
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = AmazonReviewsDataset(
        data_dir=data_dir,
        languages=languages,
        split='test',
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloaders
    def collate_fn(batch):
        """Custom collate function to stack tensors."""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long),
            'review_ids': [item['review_id'] for item in batch],
            'product_ids': [item['product_id'] for item in batch]
        }
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    # Test the preprocessing
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=".",
        languages=["en"],
        tokenizer=tokenizer,
        batch_size=8
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a sample batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch keys: {sample_batch.keys()}")
    print(f"Input shape: {sample_batch['input_ids'].shape}")
    print(f"Labels: {sample_batch['labels']}")

