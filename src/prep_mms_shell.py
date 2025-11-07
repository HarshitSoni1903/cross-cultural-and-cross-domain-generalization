#!/usr/bin/env python3
# prep_mms_shell.py
# Run via: python prep_mms_shell.py --mms_raw_dir ./data/mms --langs en es fr ja --out_root ./data
import os, re, argparse
from collections import Counter
import numpy as np
from datasets import load_from_disk, DatasetDict

LBL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    -1: 0,
    0: 1,
    1: 2,
    "neg": 0,
    "neu": 1,
    "pos": 2,
}
ID2LBL = {0: "negative", 1: "neutral", 2: "positive"}


def clean(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.replace("\u200b", "")).strip()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def stratified_split(ds, seed=42):
    n = len(ds)
    idx = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(0.8 * n), int(0.9 * n)
    return DatasetDict(
        {
            "train": ds.select(idx[:n_tr]),
            "validation": ds.select(idx[n_tr:n_va]),
            "test": ds.select(idx[n_va:]),
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mms_raw_dir", default="./data/mms")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "fr", "ja"])
    ap.add_argument("--out_root", default="./data")
    args = ap.parse_args()

    LANGS = set([l.lower() for l in args.langs])
    ensure_dir(args.out_root)
    mms = load_from_disk(args.mms_raw_dir)
    mms_ds = mms["train"] if isinstance(mms, DatasetDict) else mms

    def add_fields(ex):
        lang = (ex.get("language") or "").lower()
        ex["language_low"] = lang
        ex["text_clean"] = clean(ex.get("text", ""))
        raw = ex.get("label")
        y = LBL2ID.get(raw, LBL2ID.get(str(raw).lower(), None))
        ex["label"] = y
        ex["label_text"] = ID2LBL.get(y)
        return ex

    mms_keep = mms_ds.filter(lambda r: (r.get("language") or "").lower() in LANGS)
    mms_aug = mms_keep.map(add_fields)
    mms_aug = mms_aug.filter(
        lambda r: r["label"] in (0, 1, 2) and len(r["text_clean"]) > 0
    )
    mms_split = stratified_split(mms_aug)

    OUT_FULL = os.path.join(args.out_root, "mms_full_ja_en_fr_es")
    OUT_TRIM = os.path.join(args.out_root, "mms_tidy_ja_en_fr_es")
    ensure_dir(OUT_FULL)
    ensure_dir(OUT_TRIM)
    mms_split.save_to_disk(OUT_FULL)

    def trim(ds):
        keep = [
            c
            for c in [
                "text_clean",
                "label",
                "label_text",
                "language_low",
                "domain",
                "original_dataset",
            ]
            if c in ds.column_names
        ]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        if "language_low" in ds.column_names:
            ds = ds.rename_column("language_low", "language")
        if "text_clean" in ds.column_names:
            ds = ds.rename_column("text_clean", "text")
        return ds

    mms_trim = DatasetDict({sp: trim(ds) for sp, ds in mms_split.items()})
    mms_trim.save_to_disk(OUT_TRIM)

    for name, dsd in [("FULL", mms_split), ("TRIM", mms_trim)]:
        print(f"\n{name}:")
        for sp, d in dsd.items():
            print(f"  {sp:>10}: {len(d):,}")
    print("\n✅ MMS saved successfully")


if __name__ == "__main__":
    main()
