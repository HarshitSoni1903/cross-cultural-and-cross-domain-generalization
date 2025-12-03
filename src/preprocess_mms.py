import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

lang = "ja"
DATA_DIR = Path(f"data-translated/amazon_reviews_multi/{lang}")

df = pd.read_json(DATA_DIR / f"{lang}.jsonl", lines=True)

print(df.head(2))

keep_cols = ["_id", "text", "label", "domain", "language", "text_en"]
df["title"] = ""
df["title_en"] = ""
neg = (df['label'] == 0).sum()
neu = (df['label'] == 1).sum()
pos = (df['label'] == 2).sum()

label_map = {
    2: 5,
    1: 3,
    0: 1
}

df["label"] = df["label"].map(label_map)

neg1 = (df['label'] == 1).sum()
neu1 = (df['label'] == 3).sum()
pos1 = (df['label'] == 5).sum()

print("Before")
print(neg,pos,neu)
print("Now")
print(neg1,pos1,neu1)

assert neg == neg1, "neg != neg1"
assert neu == neu1, "neu != neu1"
assert pos == pos1, "pos != pos1"

print("mapping done successfully")

print("modifying columns")
df = df.rename(columns={"text": "review_body","text_en": "review_body_en","label": "stars"})

print("Processed DF")

print(df.head(2))

print("Splitting")
df["strat_key"] = df["stars"].astype(str) + "_" + df["domain"].astype(str)

train_df, temp_df = train_test_split(
    df,
    test_size=0.05,
    random_state=42,
    stratify=df["strat_key"]
)
print("Train")
tl=len(train_df)
print(tl)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["strat_key"]
)

print("Val")
vl=len(val_df)
print(vl)
print("Test")
tel=len(test_df)
print(tel)


for x in (train_df, val_df, test_df):
    x.drop(columns=["strat_key"], inplace=True)

print("\nTrain distribution:")
print(train_df.groupby(["stars", "domain"]).size())

print("\nVal distribution:")
print(val_df.groupby(["stars", "domain"]).size())

print("\nTest distribution:")
print(test_df.groupby(["stars", "domain"]).size())

assert vl+tl+tel == len(df), "Split sizes do not match!"

print("Splitting Completed. Now Saving")

train_df.to_json(DATA_DIR / "train.jsonl", orient="records", lines=True, force_ascii=False)
val_df.to_json(DATA_DIR / "val.jsonl", orient="records", lines=True, force_ascii=False)
test_df.to_json(DATA_DIR / "test.jsonl", orient="records", lines=True, force_ascii=False)