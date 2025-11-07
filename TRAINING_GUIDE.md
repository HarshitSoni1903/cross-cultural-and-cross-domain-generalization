# Training Guide

This guide explains how to configure and run training for different scenarios.

## Quick Start

The main training command is:
```bash
python src/train.py --config config/train_config.yaml
```

## Configuration Options

Edit `config/train_config.yaml` to customize your training:

### 1. Task Type (Classification vs Regression)

```yaml
model:
  task_type: "classification"  # or "regression"
```

- **Classification**: 3 classes (Negative=0, Neutral=1, Positive=2), uses Cross-Entropy loss
- **Regression**: Continuous ratings 1-5, uses MSE loss

### 2. Text Source (Original vs Translated)

```yaml
data:
  use_translation: false  # true or false
```

- **false**: Use original text (`review_body`) from specified languages
- **true**: Use English translations (`review_body_en`) from **ALL** available languages automatically

### 3. Languages

```yaml
data:
  train_languages: ["en"]  # Only used when use_translation=false
```

- When `use_translation=false`: Specify which languages to use (e.g., `["en"]`, `["fr"]`, `["en", "fr"]`)
- When `use_translation=true`: This setting is ignored - all available languages are used automatically

## Training Scenarios

### Scenario 1: Classification on Original English Text

```yaml
model:
  task_type: "classification"
data:
  train_languages: ["en"]
  use_translation: false
```

**Run:**
```bash
python src/train.py --config config/train_config.yaml
# or
bash train_en.sh
```

### Scenario 2: Classification on Original French Text

```yaml
model:
  task_type: "classification"
data:
  train_languages: ["fr"]
  use_translation: false
```

**Run:**
```bash
python src/train.py --config config/train_config.yaml
# or
bash train_fr.sh
```

### Scenario 3: Classification on Original English + French Text

```yaml
model:
  task_type: "classification"
data:
  train_languages: ["en", "fr"]
  use_translation: false
```

**Run:**
```bash
python src/train.py --config config/train_config.yaml
# or
bash train_en_fr.sh
```

### Scenario 4: Classification on Translated Text (All Languages)

This uses English translations from ALL available languages in `data/amazon_review/`.

```yaml
model:
  task_type: "classification"
data:
  train_languages: ["en"]  # Ignored when use_translation=true
  use_translation: true
```

**Run:**
```bash
python src/train.py --config config/train_config.yaml
```

### Scenario 5: Regression on Original Text

```yaml
model:
  task_type: "regression"
data:
  train_languages: ["en"]  # or ["fr"], ["en", "fr"], etc.
  use_translation: false
```

**Run:**
```bash
python src/train.py --config config/train_config.yaml
```

### Scenario 6: Regression on Translated Text (All Languages)

```yaml
model:
  task_type: "regression"
data:
  train_languages: ["en"]  # Ignored when use_translation=true
  use_translation: true
```

**Run:**
```bash
python src/train.py --config config/train_config.yaml
```

## Available Training Scripts

1. **train_en.sh**: Trains on English data (update config first)
2. **train_fr.sh**: Trains on French data (update config first)
3. **train_en_fr.sh**: Trains on English + French data (update config first)

**Note**: These scripts just run the training command. You still need to update the config file for your desired settings.

## Data Requirements

The training script expects data in:
```
data/amazon_review/
  ├── {language}/
  │   ├── train.jsonl          (required)
  │   ├── validation.jsonl     (optional, falls back to train if missing)
  │   └── test.jsonl           (optional, falls back to train if missing)
```

## Output

Models are saved to:
```
outputs/{task_type}_{trans/orig}_{languages}_{timestamp}/checkpoint-epoch-{epoch}/
```

Examples:
- `outputs/classification_orig_en_20241104_123456/checkpoint-epoch-1/`
- `outputs/regression_trans_all_20241104_123456/checkpoint-epoch-1/`

## Example: Complete Training Workflow

### 1. Classification on French original text:

Edit `config/train_config.yaml`:
```yaml
model:
  task_type: "classification"
data:
  train_languages: ["fr"]
  use_translation: false
```

Run:
```bash
python src/train.py --config config/train_config.yaml
```

### 2. Classification on all languages' English translations:

Edit `config/train_config.yaml`:
```yaml
model:
  task_type: "classification"
data:
  use_translation: true
```

Run:
```bash
python src/train.py --config config/train_config.yaml
```

## Tips

1. **Check available languages**: The script will automatically detect languages in `data/amazon_review/` when `use_translation=true`
2. **Validation/Test splits**: If `validation.jsonl` or `test.jsonl` files are missing, the script will use training data as fallback (with a warning)
3. **English data**: For English samples, `review_body_en` is automatically set to `review_body` (the original English text)
4. **Batch size**: Adjust `batch_size` in config if you run into memory issues
5. **Learning rate**: Default is 2e-5, adjust based on your model size and dataset

