"""
Dataset and dataloader module for Amazon review data.
Loads data from data/amazon_review/language/train.jsonl for each language.
Includes: language code, review_body, review_id, product_id, star_rating, review_body_en.
Star ratings are normalized: 1-2 → Negative (0), 3 → Neutral (1), 4-5 → Positive (2).
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class AmazonReviewDataset(Dataset):
    """
    Dataset class for Amazon reviews from data/amazon_review/language/train.jsonl.
    
    Each datapoint includes:
    - language: Language code
    - review_body: Original review text in the language
    - review_id: Unique review identifier
    - product_id: Product identifier
    - star_rating: Original star rating (1-5)
    - review_body_en: English translation of the review
    - normalized_label: Normalized label (0=Negative, 1=Neutral, 2=Positive)
    """
    
    def __init__(
        self,
        data_dir: str,
        languages: List[str],
        split: str = "train",
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        use_translation: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing amazon_review folder
            languages: List of language codes to load (e.g., ['en', 'fr'])
            split: Dataset split ('train', 'validation', 'test') - currently only 'train' is used
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length for tokenization
            use_translation: If True, use review_body_en; otherwise use review_body
        """
        self.data_dir = Path(data_dir)
        self.languages = languages
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_translation = use_translation
        self.data = []
        
        # Load data from all specified languages
        self._load_data()
    
    def _normalize_rating(self, star_rating: int) -> int:
        """
        Normalize star rating to 3-class labels.
        
        Args:
            star_rating: Original star rating (1-5)
            
        Returns:
            Normalized label: 0 (Negative), 1 (Neutral), 2 (Positive)
        """
        if star_rating <= 2:
            return 0  # Negative
        elif star_rating == 3:
            return 1  # Neutral
        else:  # 4 or 5
            return 2  # Positive
    
    def _load_data(self):
        """Load and parse JSONL files from all specified languages."""
        for lang in self.languages:
            lang_dir = self.data_dir / "amazon_review" / lang
            jsonl_file = lang_dir / f"{self.split}.jsonl"
            
            # Skip if file doesn't exist (for validation/test splits)
            if not jsonl_file.exists():
                if self.split == 'train':
                    raise FileNotFoundError(f"Training data file not found: {jsonl_file}")
                else:
                    print(f"Warning: {self.split} data file not found for language {lang}: {jsonl_file}. Skipping.")
                    continue
            
            # Load data from JSONL file
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    
                    # Extract required fields
                    language = item.get('language', lang)
                    review_body = item.get('review_body', '')
                    review_id = item.get('review_id', '')
                    product_id = item.get('product_id', '')
                    star_rating = item.get('stars', 3)
                    if star_rating == "null":
                        star_rating = 3
                    else:
                        star_rating = int(star_rating)
                    review_body_en = item.get('review_body_en', '')
                    
                    # For English samples, always use original review_body as review_body_en
                    if language == 'en' or lang == 'en':
                        review_body_en = review_body
                    
                    # Skip if essential fields are missing
                    if not review_body or not review_id or star_rating == 0:
                        continue
                    
                    # Normalize rating
                    normalized_label = self._normalize_rating(star_rating)
                    
                    self.data.append({
                        'language': language,
                        'review_body': review_body,
                        'review_id': review_id,
                        'product_id': product_id,
                        'star_rating': star_rating,
                        'review_body_en': review_body_en,
                        'normalized_label': normalized_label
                    })
        
        if len(self.data) == 0:
            if self.split == 'train':
                raise ValueError(f"No data loaded from {self.languages} for split '{self.split}'. "
                               "Please check that train.jsonl files exist for at least one language.")
            else:
                # For validation/test, it's okay to have empty dataset, but warn
                print(f"Warning: No data loaded for {self.split} split from languages {self.languages}.")
        else:
            print(f"Loaded {len(self.data)} examples from {self.languages} ({self.split})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single data sample."""
        item = self.data[idx]
        
        # Prepare output
        output = {
            'language': item['language'],
            'review_body': item['review_body'],
            'review_id': item['review_id'],
            'product_id': item['product_id'],
            'star_rating': item['star_rating'],
            'review_body_en': item['review_body_en'],
            'normalized_label': item['normalized_label'],
            'labels': item['normalized_label']  # For compatibility with training code
        }
        
        # Tokenize if tokenizer is provided
        if self.tokenizer is not None:
            # Choose text based on use_translation flag
            # Note: For dual-encoder mode, we need both texts, so use_translation is ignored here
            # The training script will handle tokenizing both separately
            text = item['review_body_en'] if self.use_translation else item['review_body']
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            output['input_ids'] = encoding['input_ids'].squeeze()
            output['attention_mask'] = encoding['attention_mask'].squeeze()
            
            # Also tokenize the other text for dual-encoder mode
            # For English reviews: use original text for both encoders
            # For non-English reviews: use translated text for frozen encoder, original text for trainable encoder
            text_original = item['review_body']
            language = item['language']
            
            if language == 'en' or language == 'en-US':
                # For English: use original text for both translated and original
                text_for_translated_encoder = text_original
            else:
                # For non-English: use translated text for frozen encoder
                text_for_translated_encoder = item['review_body_en']
            
            encoding_original = self.tokenizer(
                text_original,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            encoding_translated = self.tokenizer(
                text_for_translated_encoder,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            output['input_ids_original'] = encoding_original['input_ids'].squeeze()
            output['attention_mask_original'] = encoding_original['attention_mask'].squeeze()
            output['input_ids_translated'] = encoding_translated['input_ids'].squeeze()
            output['attention_mask_translated'] = encoding_translated['attention_mask'].squeeze()
        
        return output


def create_amazon_review_dataloaders(
    data_dir: str,
    languages: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    use_translation: bool = False,
    split: str = "train"
) -> DataLoader:
    """
    Create dataloader for Amazon review data.
    
    Args:
        data_dir: Root directory containing amazon_review folder
        languages: List of language codes (e.g., ['en', 'fr'])
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes
        use_translation: If True, use review_body_en; otherwise use review_body
        split: Dataset split ('train', 'validation', 'test') - currently only 'train' is used
        
    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = AmazonReviewDataset(
        data_dir=data_dir,
        languages=languages,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        use_translation=use_translation
    )
    
    # Create dataloader
    def collate_fn(batch):
        """Custom collate function to stack tensors and preserve metadata."""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'input_ids_original': torch.stack([item['input_ids_original'] for item in batch]),
            'attention_mask_original': torch.stack([item['attention_mask_original'] for item in batch]),
            'input_ids_translated': torch.stack([item['input_ids_translated'] for item in batch]),
            'attention_mask_translated': torch.stack([item['attention_mask_translated'] for item in batch]),
            'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long),
            'normalized_labels': torch.tensor([item['normalized_label'] for item in batch], dtype=torch.long),
            'star_ratings': torch.tensor([item['star_rating'] for item in batch], dtype=torch.long),
            'languages': [item['language'] for item in batch],
            'review_ids': [item['review_id'] for item in batch],
            'product_ids': [item['product_id'] for item in batch],
            'review_bodies': [item['review_body'] for item in batch],
            'review_bodies_en': [item['review_body_en'] for item in batch]
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader


def create_train_dataloader(
    data_dir: str,
    languages: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    use_translation: bool = False
) -> DataLoader:
    """
    Convenience function to create a training dataloader.
    
    Args:
        data_dir: Root directory containing amazon_review folder
        languages: List of language codes (e.g., ['en', 'fr'])
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes
        use_translation: If True, use review_body_en; otherwise use review_body
        
    Returns:
        Training DataLoader instance
    """
    return create_amazon_review_dataloaders(
        data_dir=data_dir,
        languages=languages,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        use_translation=use_translation,
        split='train'
    )


if __name__ == "__main__":
    # Test the dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    
    # Test loading data
    train_loader = create_train_dataloader(
        data_dir=".",
        languages=["fr"],
        tokenizer=tokenizer,
        batch_size=4,
        use_translation=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    
    # Check a sample batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch keys: {sample_batch.keys()}")
    print(f"Input shape: {sample_batch['input_ids'].shape}")
    print(f"Labels (normalized): {sample_batch['labels']}")
    print(f"Star ratings (original): {sample_batch['star_ratings']}")
    print(f"Languages: {sample_batch['languages'][:3]}")  # First 3
    print(f"Review IDs: {sample_batch['review_ids'][:3]}")  # First 3
    
    # Test with translation
    print("\n" + "="*50)
    print("Testing with translation enabled...")
    train_loader_en = create_train_dataloader(
        data_dir=".",
        languages=["fr"],
        tokenizer=tokenizer,
        batch_size=4,
        use_translation=True
    )
    
    sample_batch_en = next(iter(train_loader_en))
    print(f"Using English translations: {sample_batch_en['review_bodies_en'][0][:100]}...")

