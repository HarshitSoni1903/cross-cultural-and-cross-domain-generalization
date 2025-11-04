#!/bin/bash
# Script to train model on English + French data

echo "Training on English + French data..."
echo "Note: Update config/train_config.yaml with train_languages: [\"en\", \"fr\"]"
python src/train.py --config config/train_config.yaml

echo "Training complete!"

