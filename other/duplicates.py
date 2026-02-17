import pandas as pd

df = pd.read_csv("cars.csv")

dup_mask = df.duplicated(subset=["vin"], keep=False)
dups = df[dup_mask].sort_values("vin")

print("Viso eilučių:", len(df))
print("Unikalių VIN:", df["vin"].nunique())
print("Dublikatų kiekis:", len(df) - df["vin"].nunique())

clean_df = df.drop_duplicates(subset=["url"], keep="first")
clean_df.to_csv("cars.csv", index=False)