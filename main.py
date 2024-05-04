import pandas as pd
from preprocess_books import process_books
from preprocess_users import process_users

# process_books and process_users do not actually delete any rows.
# They just clean the current data and return rows to be deleted
books_df = pd.read_csv("COMP20008-A2-Data-Files\BX-Books.csv")
ISBN_del = process_books(df=books_df)
users_df = pd.read_csv("COMP20008-A2-Data-Files\BX-Users.csv")
USER_ID_del = process_users(df=users_df)
ratings_df = pd.read_csv("COMP20008-A2-Data-Files\BX-Ratings.csv")

# Remove Abnormal ISBN
books_df = books_df[~books_df["ISBN"].isin(ISBN_del)]
ratings_df = ratings_df[~ratings_df["ISBN"].isin(ISBN_del)]

# Remove Abnormal USER_ID
users_df = users_df[~users_df["User-ID"].isin(USER_ID_del)]
ratings_df = ratings_df[~ratings_df["User-ID"].isin(USER_ID_del)]

""""
    From books_df remove ISBN,
    From users_df remove USER_ID,
    From ratings_df remove USER_ID and ISBN
"""


books_df.to_csv("datasets_preprocessed\BX-Books_Processed.csv", index=False)
users_df.to_csv("datasets_preprocessed/BX-Users_Processed.csv", index=False)
ratings_df.to_csv("datasets_preprocessed/BX-Ratings_Processed.csv", index=False)