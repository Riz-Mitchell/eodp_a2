import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot
from sklearn.decomposition import PCA
import numpy as np

# clusters users based on their book ratings using kmeans and pca for dimensionality reduction
def kmeans_clustering(df):

    user_ratings_matrix = df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

    if user_ratings_matrix.shape[1] > 50:
        # reduce dimensionality
        pca = PCA(n_components = 50)
        data_transformed = pca.fit_transform(user_ratings_matrix)
    else:
        # normalizes features for clustering
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(user_ratings_matrix)

    # limit number of clusters
    num_clusters = min(5, len(data_transformed) - 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(data_transformed)

    user_cluster_mapping = pd.Series(clusters, index = user_ratings_matrix.index)
    
    df['Cluster'] = df['User-ID'].map(user_cluster_mapping)

    return df, data_transformed, clusters
