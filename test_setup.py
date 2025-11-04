"""
Quick test script to verify the setup is correct.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers not installed")

try:
    from src.model import XLMROBERTaCORAL
    print("Model module imported successfully")
except Exception as e:
    print(f"Error importing model: {e}")

try:
    from src.data_preprocessing import create_dataloaders
    print("Data preprocessing module imported successfully")
except Exception as e:
    print(f"Error importing data preprocessing: {e}")

print("\nSetup test complete!")

