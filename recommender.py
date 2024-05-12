import pandas as pd
from preprocessing import process_bx_books, process_bx_users, process_bx_ratings
from kmeansclustering import kmeans_clustering
import matplotlib.pyplot as plt
import seaborn as sns

def recommend_books_for_generation(clustered_df, target_generation, books_df):
    # filter the clustered data to include only rows where 'generation' matches target generation
    gen_df = clustered_df[clustered_df['Generation'] == target_generation]
    
    # determines the most common cluster(s) within the filtered dataframe
    predominant_clusters = gen_df['Cluster'].value_counts().nlargest(1).index

    # further filtering from predominant clusters
    cluster_books = gen_df[gen_df['Cluster'].isin(predominant_clusters)]
    
    # compute average rating for each book within these clusters 
    average_ratings = cluster_books.groupby('ISBN')['Book-Rating'].mean().reset_index()
    
    # filter to include books with average rating of 8.5 or higher
    high_rated_books = average_ratings[average_ratings['Book-Rating'] >= 8.5]

    high_rated_books = pd.merge(high_rated_books, books_df, on='ISBN')

    return high_rated_books[['ISBN','Book-Title','Book-Rating']]
