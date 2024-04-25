import pandas as pd

df = pd.read_csv("COMP20008-A2-Data-Files\BX-Ratings.csv")

abnormal_rows = df.loc[df["Book-Rating"] < 0]

print(abnormal_rows.head(10))