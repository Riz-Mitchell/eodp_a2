import pandas as pd
import numpy as np


def process_users(df):
    
    for index, row in df.iterrows():
        if pd.notna(row["User-State"]):
            df.at[index, "User-State"] = row["User-State"].strip()
        
        if pd.notna(row["User-Country"]):
            df.at[index, "User-Country"] = row["User-Country"].replace('"','').strip()
        
        if pd.notna(row["User-Age"]):
            if (type(row["User-Age"]) == str):
                temp = row["User-Age"].replace('"','').strip()
                df.at[index, "User-Age"] = int(temp)
    

    df['User-Age'].replace('', np.nan, inplace=True)

    USER_ID_del = df.loc[df["User-Age"].isna()]["User-ID"]
    
    return USER_ID_del