import pandas as pd
from datetime import datetime
from preprocessing import process_bx_books, process_bx_ratings, process_bx_users
from kmeansclustering import kmeans_clustering
from cluster_distribution import plot_generation_cluster_distribution, plot_2d_clusters, plot_3d_clusters, plot_pairplot
import matplotlib.pyplot as plt
import seaborn as sns
from regression_model import run_regression_model
from recommender import recommend_books_for_generation
from kmeans_evaluation import kmeans_evaluation
import json

def main():
    # Read and pre-process the training dataset
    books_df = pd.read_csv("COMP20008-A2-Data-Files/BX-Books.csv")
    books_df = process_bx_books(books_df)
    users_df = pd.read_csv("COMP20008-A2-Data-Files/BX-Users.csv")
    users_df = process_bx_users(users_df)
    ratings_df = pd.read_csv("COMP20008-A2-Data-Files/BX-Ratings.csv")
    ratings_df = process_bx_ratings(ratings_df)

    books_df.to_csv("datasets_preprocessed/BX-Books_Processed.csv", index=False)
    users_df.to_csv("datasets_preprocessed/BX-Users_Processed.csv", index=False)
    ratings_df.to_csv("datasets_preprocessed/BX-Ratings_Processed.csv", index=False)

    # Read and pre-process the testing dataset
    new_books_df = pd.read_csv("COMP20008-A2-Data-Files/BX-NewBooks.csv")
    new_books_df = process_bx_books(new_books_df)
    new_users_df = pd.read_csv("COMP20008-A2-Data-Files/BX-NewBooksUsers.csv")
    new_users_df = process_bx_users(new_users_df)
    new_ratings_df = pd.read_csv("COMP20008-A2-Data-Files/BX-NewBooksRatings.csv")
    new_ratings_df = process_bx_ratings(new_ratings_df)

    new_books_df.to_csv("datasets_preprocessed/BX-NewBooks_Processed.csv", index=False)
    new_users_df.to_csv("datasets_preprocessed/BX-NewUsers_Processed.csv", index=False)
    new_ratings_df.to_csv("datasets_preprocessed/BX-NewRatings_Processed.csv", index=False)

    # Merge the testing dataset
    merged_df = pd.merge(pd.merge(ratings_df, users_df, on="User-ID"), books_df, on="ISBN")

    # Merge the training dataset
    new_merged_df = pd.merge(pd.merge(new_ratings_df, new_users_df, on='User-ID'), new_books_df, on='ISBN')

    new_clustered_df, data_transformed, clusters = kmeans_clustering(new_merged_df)

    # Regression Part: Run Regression Model
    run_regression_model(merged_df, new_merged_df)

    # Clustering Part: Training
    clustered_df = kmeans_clustering(merged_df)[0]
    clustered_df.to_csv("datasets_preprocessed/Clustered_Data.csv", index=False)

    # New dtesting
    kmeans_evaluation(data_transformed, clusters, new_clustered_df)

    plot_2d_clusters(clustered_df, 'Cluster')
    plot_3d_clusters(clustered_df, 'Cluster')
    plot_pairplot(clustered_df, 'Cluster')
    plot_generation_cluster_distribution(clustered_df)

    target_generation = input("Enter your generation (e.g., Millennials, Generation X): ")
    recommendations = recommend_books_for_generation(clustered_df, target_generation, books_df)

    print("Top 5 recommended books:")
    print(recommendations.head())

    recommendations.to_csv("recommended_books.csv", index=False)
    print("All recommendations have been saved to 'recommended_books.csv'.")

main()
