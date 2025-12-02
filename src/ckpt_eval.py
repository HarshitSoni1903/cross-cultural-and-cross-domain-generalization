"""Usage:
python src/ckpt_eval.py --checkpoint_dir outputs/baseline_temp/classification_orig_en_zh_20251202_024035/checkpoint-epoch-1

The dir already has checkpoints as:

outputs/NLND_temp/classification_trans_en_zh_20251202_023752/
│
├── config.yaml
├── test_results.json
├── log/
└── checkpoint-epoch-1/
"""
import argparse
import yaml
import torch
import json
import numpy as np
from pathlib import Path

from transformers import AutoTokenizer
from amazon_review_dataset import create_amazon_review_dataloaders
from model import XLMROBERTaRating, DualEncoderXLMROBERTaRating
from train import Trainer
from eval_model import run_tsne_umap


def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    return obj


def load_model_from_checkpoint(config, checkpoint_dir, device):
    training_schema = config["model"].get("training_schema", "single")
    task_type = config["model"].get("task_type", "regression")
    num_classes = 3 if task_type == "classification" else config["model"].get("num_labels", 5)
    base_model = config["model"]["base_model"]

    if training_schema == "dual_encoder":
        model = DualEncoderXLMROBERTaRating(
            pretrained_encoder_path=config["model"]["pretrained_encoder_path"],
            base_model_name=base_model,
            num_classes=num_classes,
            freeze_pretrained=True,
            baseline_checkpoint_path=config["model"].get("baseline_checkpoint_path"),
            classifier_fusion_method=config["model"].get("classifier_fusion_method", "concat"),
            nlnd_drop_prob=float(config["model"].get("nlnd_drop_prob", 0.0)),
            use_ld_masking=bool(config["model"].get("use_ld_masking", False)),
        )
    else:
        model = XLMROBERTaRating(
            model_name=base_model,
            num_classes=num_classes,
            task_type=task_type,
        )

    model = model.from_pretrained(str(checkpoint_dir)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    return model, tokenizer


def build_test_loader(config, tokenizer):
    return create_amazon_review_dataloaders(
        data_dir=config["data"]["data_path"],
        languages=config["data"]["train_languages"],
        tokenizer=tokenizer,
        max_length=config["training"]["max_length"],
        batch_size=config["training"]["batch_size"],
        use_translation=config["data"].get("use_translation", False),
        split="test",
        domain_info=config["data"].get("domain_info", False),
        training_schema=config["model"].get("training_schema", "single"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir).resolve()
    exp_root = ckpt.parent

    config_path = exp_root / "config_used.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_prefix = exp_root.parent.name
    outputs_dir = exp_root.parent.parent
    eval_prefix = f"{run_prefix}_evals"

    eval_root = outputs_dir / eval_prefix / exp_root.name / ckpt.name
    eval_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_from_checkpoint(config, ckpt, device)
    test_loader = build_test_loader(config, tokenizer)

    temp_trainer = Trainer(config)
    temp_trainer.model = model
    temp_trainer.tokenizer = tokenizer
    temp_trainer.test_loader = test_loader

    metrics = temp_trainer.evaluate(test_loader, split_name="Test", epoch=None)

    results = {
        "checkpoint_dir": str(ckpt),
        "eval_output_dir": str(eval_root),
        "training_schema": config["model"].get("training_schema"),
        "task_type": config["model"].get("task_type"),
        "languages": config["data"]["train_languages"],
        "use_translation": config["data"].get("use_translation"),
        "test_samples": len(test_loader.dataset),
        "metrics": make_json_serializable(metrics),
    }

    with open(eval_root / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    run_tsne_umap(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_loader,
        device=device,
        output_dir=eval_root,
        max_points=config.get("evaluation", {}).get("max_points", 10000),
    )


if __name__ == "__main__":
    main()
