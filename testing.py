import regex
import pandas as pd
import csv

df = pd.read_csv("COMP20008-A2-Data-Files\BX-Users.csv")

print(f"Dataset before altering: {len(df)}")
# empty_country_rows = df[df['User-Country'].isnull()]

empty_age_rows = df[df['User-Age'].isnull()]

# clean_text = lambda cell: cell.strip('"') if isinstance(cell,str)

# df["User-Country"] = df["User-Country"].apply(clean_text)

# print(empty_country_rows)

print(empty_age_rows)

df.to_csv("datasets_preprocessed/BX-Users_Processed.csv", index=False)