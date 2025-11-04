# Cross-Lingual Sentiment Analysis with CORAL

A complete implementation of cross-lingual sentiment analysis using XLM-RoBERTa base model with a CORAL (Consistent Rank Logits) ordinal regression head for predicting star ratings (1-5) from Amazon multilingual reviews.

## Project Structure

```
DSGA-1011-PROJECT/
├── config/
│   └── train_config.yaml          # Training hyperparameters configuration
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── model.py                   # XLM-RoBERTa + CORAL head architecture
│   ├── train.py                   # Training script
│   └── inference.py               # Inference module
├── requirements.txt               # Python dependencies
├── amazon_reviews_multi/          # Amazon multilingual reviews dataset
│   ├── en/                        # English reviews
│   ├── fr/                        # French reviews
│   ├── de/                        # German reviews
│   ├── es/                        # Spanish reviews
│   ├── ja/                        # Japanese reviews
│   └── zh/                        # Chinese reviews
└── README.md                      # This file
```

## Installation

1. **Install Python 3.11** (or compatible version)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.0+ with CUDA support
- Transformers 4.30+
- Datasets 2.12+
- PyYAML 6.0+
- NumPy, scikit-learn, tqdm, and accelerate

## Dataset

The project uses the Amazon Multilingual Reviews dataset from the Hugging Face dataset repository. The data is already included in the `amazon_reviews_multi/` directory with the following structure:

- Each language has `train.jsonl.gz`, `validation.jsonl.gz`, and `test.jsonl.gz` files
- Each review contains: `review_id`, `product_id`, `reviewer_id`, `stars` (1-5), `review_body`, `review_title`, `language`, and `product_category`
- Supported languages: English (en), French (fr), German (de), Spanish (es), Japanese (ja), Chinese (zh)

## Configuration

Edit `config/train_config.yaml` to customize training parameters:

```yaml
model:
  base_model: "FacebookAI/xlm-roberta-base"
  num_labels: 5
  
training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 1
  warmup_steps: 500
  weight_decay: 0.01
  max_length: 512
  
data:
  train_languages: ["en"]  # Change to ["fr"], ["en", "fr"], etc.
  output_dir: "./outputs"
  save_steps: 1000
  eval_steps: 500
  logging_steps: 100
```

## Training

### Training Workflow

The project implements a three-phase training strategy:

1. **Phase 1**: Train on English data only → `model_en/`
2. **Phase 2**: Train on French data only → `model_fr/`
3. **Phase 3**: Train on combined English + French data → `model_en_fr/`

### Running Training

**Train on English data**:
```bash
# Update config/train_config.yaml: train_languages: ["en"]
python src/train.py --config config/train_config.yaml
```

**Train on French data**:
```bash
# Update config/train_config.yaml: train_languages: ["fr"]
python src/train.py --config config/train_config.yaml
```

**Train on combined English + French data**:
```bash
# Update config/train_config.yaml: train_languages: ["en", "fr"]
python src/train.py --config config/train_config.yaml
```

### Training Output

During training, you'll see:
- Training loss and learning rate updates
- Validation metrics (loss, MAE, accuracy, per-class accuracy)
- Model checkpoints saved to `outputs/checkpoint-epoch-X/`

## Inference

### Single Review Prediction

```bash
python src/inference.py \
    --model_path outputs/checkpoint-epoch-1 \
    --title "Great product!" \
    --body "I love this product. It works perfectly!" \
    --probs
```

### Batch Inference

```bash
python src/inference.py \
    --model_path outputs/checkpoint-epoch-1
```

This will run example predictions and display results.

### Using the Predictor in Python

```python
from src.inference import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor("outputs/checkpoint-epoch-1")

# Predict single review
result = predictor.predict_single(
    review_title="Amazing quality",
    review_body="This product exceeded my expectations...",
    return_probs=True
)

print(f"Predicted Stars: {result['prediction']}/5")
print(f"Probabilities: {result['probabilities']}")
```

## Model Architecture

### XLM-RoBERTa Base
- **Model**: `FacebookAI/xlm-roberta-base`
- **Hidden size**: 768
- **Parameters**: 278M
- **Multilingual**: Pre-trained on 100 languages

### CORAL Ordinal Regression Head
- **Input**: 768-dimensional embeddings from [CLS] token
- **Output**: 4 cumulative logits (for 5 classes, representing thresholds between classes)
- **Final prediction**: Mapped to [1, 5] star range
- **Loss**: Binary cross-entropy on cumulative logits (CORAL loss)

### Why CORAL?

CORAL (Consistent Rank Logits) is designed for ordinal regression tasks where classes have a natural ordering (1 < 2 < 3 < 4 < 5 stars). Unlike standard classification, CORAL:
- Models cumulative probabilities P(y ≥ j)
- Ensures predicted ratings respect the ordinal relationship
- Typically outperforms standard classification for ordinal problems

## Evaluation Metrics

The model reports:
- **MAE (Mean Absolute Error)**: Average difference between predicted and true star ratings
- **Accuracy**: Exact match accuracy (% of exactly correct predictions)
- **Per-class Accuracy**: Accuracy for each star rating (1-5)

## Example Usage

### Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train on English data:
```bash
python src/train.py --config config/train_config.yaml
```

3. Run inference:
```bash
python src/inference.py --model_path outputs/checkpoint-epoch-1 --title "Good" --body "Nice product"
```

## Extending to Other Languages

The infrastructure supports all languages in the dataset (de, es, ja, zh). To train on other languages:

1. Edit `config/train_config.yaml`:
```yaml
data:
  train_languages: ["de"]  # German
  # or ["es"], ["ja"], ["zh"], etc.
```

2. Run training:
```bash
python src/train.py --config config/train_config.yaml
```

## Technical Details

- **Framework**: PyTorch 2.0+
- **Transformers**: Hugging Face Transformers 4.30+
- **CUDA Support**: Automatic GPU detection and usage
- **Optimization**: AdamW optimizer with linear warmup scheduler
- **Tokenization**: Max length 512, padding and truncation
- **Memory Efficient**: Uses gradient accumulation if needed

## License

This project is for educational and research purposes. Please check the licenses of:
- Amazon Reviews dataset
- XLM-RoBERTa model (Facebook/Meta AI)

## References

- [XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)
- [CORAL Paper](https://arxiv.org/abs/1901.07884)
- [Amazon Multilingual Reviews Dataset](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi)

