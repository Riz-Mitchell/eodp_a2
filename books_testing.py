import pandas as pd
import numpy as np
import re

df = pd.read_csv("COMP20008-A2-Data-Files\BX-Books.csv")

print(df.head(10))

yr_of_pub_series = df.loc[df["Year-Of-Publication"] < 1000]

print(yr_of_pub_series.head(10))
print(f"Length of PUB YEAR series is: {len(yr_of_pub_series)}")

pattern = r"-\d+"

ISBN_series = df.loc[df["ISBN"].str.contains(pattern, regex=True)]

print(ISBN_series.head(10))
# print(f"Length of ISBN series is: {len(ISBN_series)}")

    