#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, gzip, logging, torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

LANG_CODES = {"ja": "jpn_Jpan", "fr": "fra_Latn", "zh": "zho_Hans"}
LOGGER = logging.getLogger("translate_fast_resume")

# ---------- logging ----------
def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)s - %(message)s"
    logging.basicConfig(level=lvl, format=fmt, datefmt="%H:%M:%S")
    logging.getLogger("transformers").setLevel(max(lvl, logging.WARNING))
    logging.getLogger("huggingface_hub").setLevel(max(lvl, logging.WARNING))

# ---------- utils ----------
def ensure_abs(path: str) -> str:
    return os.path.abspath(path)

def ensure_model(model_dir: str, hf_repo_id: str, hf_token: Optional[str]) -> None:
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        LOGGER.info("Model directory already contains config.json, skipping download: %s", model_dir)
        return
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub not available; cannot download model.")
    if not hf_repo_id:
        raise RuntimeError("hf_repo_id required for download.")
    if not hf_token:
        raise RuntimeError("HF token required to download NLLB model.")
    os.makedirs(model_dir, exist_ok=True)
    LOGGER.info("Downloading model snapshot %s -> %s", hf_repo_id, model_dir)
    snapshot_download(repo_id=hf_repo_id,
                      local_dir=model_dir,
                      local_dir_use_symlinks=False,
                      token=hf_token)
    LOGGER.info("Model snapshot downloaded.")

# ---------- load model ----------
def load_model(model_dir: str, dtype: str, device_map: str):
    if dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    LOGGER.info("Loading model from %s (dtype=%s, device_map=%s)", model_dir, torch_dtype, device_map)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    mdl.eval()
    LOGGER.info("Model loaded.")
    return tok, mdl

# ---------- translation ----------
@torch.inference_mode()
def translate_batch(tokenizer, model, texts: List[str], src_lang: str,
                    max_new_tokens: int, beam_size: int) -> List[str]:
    """Batch translation using NLLB generation."""
    if not texts:
        return []

    device = next(model.parameters()).device
    tokenizer.src_lang = LANG_CODES.get(src_lang, src_lang)
    tgt_code = "eng_Latn"
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_code)
    if tgt_id is None:
        tgt_id = tokenizer.lang_code_to_id.get(tgt_code)

    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    gen = model.generate(
        **inputs,
        forced_bos_token_id=tgt_id,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        do_sample=False,
    )
    outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [o.strip() for o in outs]

# ---------- process file ----------
def process_file(src_path: str, dst_path: str, tokenizer, model, args, lang_hint: Optional[str]):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    LOGGER.info("Processing %s -> %s (lang=%s)", src_path, dst_path, lang_hint)

    done = 0
    if args.resume and os.path.exists(dst_path):
        with open(dst_path, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        LOGGER.info("Resuming: %d lines already translated.", done)

    total, translated = 0, 0
    batch_texts, batch_objs = [], []

    opener = gzip.open if src_path.endswith(".gz") else open
    
    with opener(src_path, "rt", encoding="utf-8") as src, open(dst_path, "a", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            if total <= done:
                continue  # skip previously done lines

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed line %d", total)
                continue

            text = obj.get(args.text_column, "")
            if not isinstance(text, str):
                text = ""
            batch_objs.append(obj)
            batch_texts.append(text)

            if len(batch_objs) >= args.batch_size:
                outs = translate_batch(tokenizer, model, batch_texts, lang_hint,
                                       args.max_new_tokens, args.beam_size)
                for o, tr in zip(batch_objs, outs):
                    o[args.output_column] = tr
                    dst.write(json.dumps(o, ensure_ascii=False) + "\n")
                dst.flush()
                translated += len(batch_objs)
                batch_objs, batch_texts = [], []

                if translated % (args.print_every * args.batch_size) == 0:
                    LOGGER.info("Progress: %d lines translated", translated + done)

        # remaining batch
        if batch_objs:
            outs = translate_batch(tokenizer, model, batch_texts, lang_hint,
                                   args.max_new_tokens, args.beam_size)
            for o, tr in zip(batch_objs, outs):
                o[args.output_column] = tr
                dst.write(json.dumps(o, ensure_ascii=False) + "\n")
            dst.flush()
            translated += len(batch_objs)

    LOGGER.info("Completed %s | total=%d | newly_translated=%d", src_path, total, translated)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--languages", nargs="+", default=["ja", "fr", "zh"])
    ap.add_argument("--begin", type=int, default=0,
                    help="Start index for language processing (for parallel jobs)")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--hf_repo_id", default="facebook/nllb-200-3.3B")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--text_column", default="review_body")
    ap.add_argument("--output_column", default="review_body_en")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--beam_size", type=int, default=2)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--log_level", default="INFO")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--print_every", type=int, default=50)
    args = ap.parse_args()

    setup_logging(args.log_level)
    args.data_dir = ensure_abs(args.data_dir)
    args.output_root = ensure_abs(args.output_root)
    args.model_dir = ensure_abs(args.model_dir)

    ensure_model(args.model_dir, args.hf_repo_id, args.hf_token or os.environ.get("HF_TOKEN"))
    tok, mdl = load_model(args.model_dir, args.dtype, args.device_map)

    langs = args.languages[args.begin:]
    for lang in langs:
        break
        for split in args.splits:
            rel = os.path.join("amazon_reviews_multi", lang, f"{split}.jsonl.gz")
            src_path = os.path.join(args.data_dir, rel)
            dst_path = os.path.join(args.output_root, "amazon_reviews_multi", lang, f"{split}.jsonl")
            if not os.path.exists(src_path):
                LOGGER.warning("Missing: %s", src_path)
                continue
            process_file(src_path, dst_path, tok, mdl, args, lang)

    LOGGER.info("Review Body Translation Complete")
    
    LOGGER.info("Review Title Translation")
    
    for lang in langs:
        for split in args.splits:
            src_path = os.path.join(args.output_root, "amazon_reviews_multi", lang, f"{split}.jsonl")
            dst_path = os.path.join(
                args.output_root, "amazon_reviews_multi", lang, f"{split}.titles.jsonl"
            )

            if not os.path.exists(src_path):
                LOGGER.warning("Missing translated file: %s", src_path)
                continue

            # Override only for title translation
            args.text_column = "review_title"
            args.output_column = "review_title_en"

            process_file(src_path, dst_path, tok, mdl, args, lang)
    
    
    LOGGER.info("All done.")

if __name__ == "__main__":
    main()