import pandas as pd
import regex

df = pd.read_csv("COMP20008-A2-Data-Files\BX-Books.csv")
# print(df.head(40))

# Targets brackets and brackets within brackets
br_pattern = r"\((?:[^()]++|(?R))*+\)"
non_ASCII_pattern = regex.compile(r"[^\x00-\x7F]")

clean_text = lambda cell: regex.sub(br_pattern, '', cell).strip().replace('"', '')
non_ASCII = lambda cell: bool(non_ASCII_pattern.search(cell))



df["Book-Title"] = df["Book-Title"].apply(clean_text)
df["Book-Publisher"] = df["Book-Publisher"].apply(clean_text)

# Select rows with non english characters
non_english_rows_BT = df[df["Book-Title"].apply(non_ASCII)]
non_english_rows_BA = df[df["Book-Author"].apply(non_ASCII)]
non_english_rows_BP = df[df["Book-Publisher"].apply(non_ASCII)]

rows_to_remove = pd.concat([non_english_rows_BT, non_english_rows_BA, non_english_rows_BP], ignore_index=True)

rows_to_remove = rows_to_remove.drop_duplicates(["ISBN"], keep="first")

ISBN_del = rows_to_remove["ISBN"]

print(ISBN_del)
    
df.to_csv("datasets_preprocessed/BX-Books_Processed.csv", index=False)