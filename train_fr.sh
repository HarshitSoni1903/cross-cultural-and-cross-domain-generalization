#!/bin/bash
# Script to train model on French data

echo "Training on French data..."
echo "Note: Update config/train_config.yaml with train_languages: [\"fr\"]"
python src/train.py --config config/train_config.yaml

echo "Training complete!"

