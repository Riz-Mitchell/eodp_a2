import regex
from datetime import datetime
import numpy as np
import pandas as pd

# Assigns a generation to each book based on year of publications
def assign_book_generation(year):
    if 1928 <= year <= 1945:
        return "Silent Generation"
    elif 1946 <= year <= 1964:
        return "Baby Boomers"
    elif 1965 <= year <= 1980:
        return "Generation X"
    elif 1981 <= year <= 1996:
        return "Millennials"
    elif 1997 <= year <= 2012:
        return "Generation Z"
    else:
        return "Other"

# Cleans data using regex and dropping any rows with empty columns
# assigns generational labels 
def process_bx_books(df):

    br_pattern = r"\((?:[^()]++|(?R))*+\)"

    non_ASCII_pattern = regex.compile(r"[^\x00-\x7F]")

    clean_text = lambda cell: regex.sub(br_pattern, '', cell).strip().replace('"', '').lower()

    non_ASCII = lambda cell: bool(non_ASCII_pattern.search(cell))

    df["Book-Title"] = df["Book-Title"].apply(clean_text)
    df["Book-Author"] = df["Book-Author"].apply(clean_text)
    df["Book-Publisher"] = df["Book-Publisher"].apply(clean_text)

    non_english_rows = df[df.apply(lambda row: any(non_ASCII(row[col]) for col in ["Book-Title", "Book-Author", "Book-Publisher"]), axis=1)]
    df = df.drop(non_english_rows.index)

    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    
    df = df[df['Year-Of-Publication'].notna() & df['Year-Of-Publication'].between(1900, datetime.now().year)].copy()

    df["Book-Generation"] = df["Year-Of-Publication"].apply(assign_book_generation)
    
    return df
# Cleans data using regex and dropping any rows with empty columns
# categorizes by generation based on age 
def process_bx_users(df):
    
    df = df.copy()  
    df["User-State"] = df["User-State"].apply(lambda x: regex.sub(r"[^\w\s]", '', x).strip().lower() if pd.notna(x) else x)
    df["User-Country"] = df["User-Country"].apply(lambda x: x.replace('"', '').strip().lower() if pd.notna(x) else x)

    df["User-Age"] = pd.to_numeric(df["User-Age"], errors='coerce')

    df = df[(df["User-Age"] >= 10) & (df["User-Age"] <= 100)]
    
    df = df.dropna()

    df['Birth-Year'] = 2024 - df['User-Age']

    bins = [0, 1928, 1945, 1964, 1980, 1996, 2012, np.inf]
    labels = ["Unknown", "Silent Generation", "Baby Boomers", "Generation X", "Millennials", "Generation Z", "Generation Alpha"]
    
    df['Generation'] = pd.cut(df['Birth-Year'], bins=bins, labels=labels, right=True)

    return df

# cleans and filters rating data
def process_bx_ratings(df):
    
    df['Book-Rating'] = pd.to_numeric(df['Book-Rating'], errors='coerce')
    df = df[df['Book-Rating'].between(1, 10)].copy()
    return df

