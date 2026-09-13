import json
import pandas as pd
with open("data/processed/splits.json") as f:
    splits = json.load(f)
val_ids = splits["val"]
df = pd.read_csv("data/processed/processed.csv")
val_df = df[df["id"].isin(val_ids)]
print(len(val_df))
