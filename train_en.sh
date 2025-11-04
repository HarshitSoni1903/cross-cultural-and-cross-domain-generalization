#!/bin/bash
# Script to train model on English data

echo "Training on English data..."
python src/train.py --config config/train_config.yaml

echo "Training complete!"

