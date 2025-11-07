#!/usr/bin/env python3
# prep_amazon_shell.py
# Run via: python prep_amazon_shell.py --amazon_raw_dir ./data/amazon_reviews/raw --langs en es fr ja --out_root ./data
import os, re, argparse
from datasets import load_from_disk, load_dataset, DatasetDict


def clean(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.replace("\u200b", "")).strip()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def stars_to_label(stars):
    try:
        stars = int(stars)
    except:
        stars = 3
    if stars <= 2:
        return 0, "negative"
    if stars == 3:
        return 1, "neutral"
    return 2, "positive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amazon_raw_dir", default="./data/amazon_reviews/raw")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "fr", "ja"])
    ap.add_argument("--out_root", default="./data")
    args = ap.parse_args()

    LANGS = set([l.lower() for l in args.langs])
    ensure_dir(args.out_root)

    if os.path.isdir(args.amazon_raw_dir):
        amazon = load_from_disk(args.amazon_raw_dir)
    else:
        amazon = load_dataset("buruzaemon/amazon_reviews_multi", "all_languages")

    def add_fields(ex):
        title, body = ex.get("review_title") or "", ex.get("review_body") or ""
        ex["text_merged"] = clean((title + " " + body).strip())
        ex["language_low"] = (ex.get("language") or "").lower()
        y, yt = stars_to_label(ex.get("stars"))
        ex["label"], ex["label_text"] = y, yt
        return ex

    amazon_aug = DatasetDict({sp: ds.map(add_fields) for sp, ds in amazon.items()})
    amazon_aug = DatasetDict(
        {
            sp: ds.filter(
                lambda r: r["language_low"] in LANGS and len(r["text_merged"]) > 0
            )
            for sp, ds in amazon_aug.items()
        }
    )

    OUT_FULL = os.path.join(args.out_root, "amazon_full_ja_en_fr_es")
    OUT_TRIM = os.path.join(args.out_root, "amazon_tidy_ja_en_fr_es")
    ensure_dir(OUT_FULL)
    ensure_dir(OUT_TRIM)
    amazon_aug.save_to_disk(OUT_FULL)

    def trim(ds):
        keep = [
            c
            for c in [
                "text_merged",
                "label",
                "label_text",
                "language_low",
                "product_category",
            ]
            if c in ds.column_names
        ]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        if "text_merged" in ds.column_names:
            ds = ds.rename_column("text_merged", "text")
        if "language_low" in ds.column_names:
            ds = ds.rename_column("language_low", "language")
        return ds

    amazon_trim = DatasetDict({sp: trim(ds) for sp, ds in amazon_aug.items()})
    amazon_trim.save_to_disk(OUT_TRIM)

    for name, dsd in [("FULL", amazon_aug), ("TRIM", amazon_trim)]:
        print(f"\n{name}:")
        for sp, d in dsd.items():
            print(f"  {sp:>10}: {len(d):,}")
    print("\n✅ Amazon saved successfully")


if __name__ == "__main__":
    main()
