import os
import json
import random
from pathlib import Path

# ---------- CONFIG ----------
SRC_DIR = Path("data/amazon_review")
DEST_DIR = Path("data_temp/amazon_review")
TRAIN_N = 1000
VAL_N = 50
TEST_N = 50

# Make destination directory
DEST_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    """Load a .jsonl file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(path, items):
    """Write list of dicts to .jsonl."""
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------- MAIN LOOP ----------
for lang_dir in SRC_DIR.iterdir():
    if not lang_dir.is_dir():
        continue
    
    lang = lang_dir.name
    print(f"Processing language: {lang}")
    
    # create destination directory for this language
    out_lang_dir = DEST_DIR / lang
    out_lang_dir.mkdir(parents=True, exist_ok=True)
    
    # define source files
    train_path = lang_dir / "train.jsonl"
    val_path = lang_dir / "validation.jsonl"
    test_path = lang_dir / "test.jsonl"

    # load with fallback if missing
    if train_path.exists():
        train = load_jsonl(train_path)
        train_sample = random.sample(train, min(TRAIN_N, len(train)))
        write_jsonl(out_lang_dir / "train.jsonl", train_sample)
        print(f"  train: {len(train)} → {len(train_sample)}")
    else:
        print(f"  train missing for {lang}")

    if val_path.exists():
        val = load_jsonl(val_path)
        val_sample = random.sample(val, min(VAL_N, len(val)))
        write_jsonl(out_lang_dir / "validation.jsonl", val_sample)
        print(f"  validation: {len(val)} → {len(val_sample)}")
    else:
        print(f"  validation missing for {lang}")

    if test_path.exists():
        test = load_jsonl(test_path)
        test_sample = random.sample(test, min(TEST_N, len(test)))
        write_jsonl(out_lang_dir / "test.jsonl", test_sample)
        print(f"  test: {len(test)} → {len(test_sample)}")
    else:
        print(f"  test missing for {lang}")

print("\nDone. Mini dataset created in data_temp/amazon_review/")
