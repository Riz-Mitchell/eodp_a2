import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np

# Function to plot the distribution of generations within clusters
def plot_generation_cluster_distribution(clustered_df):
    # Create a crosstab to see how many users of each generation fall into each cluster
    generation_cluster = pd.crosstab(clustered_df['Generation'], clustered_df['Cluster'])

    plt.figure(figsize=(10, 7))
    generation_cluster.plot(kind='bar', stacked=True)
    plt.title('Distribution of Generations within Clusters')
    plt.xlabel('Generation')
    plt.ylabel('Number of Users')
    plt.legend(title='Cluster')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cluster_visualisations/generation_cluster_distribution.png')
    plt.close()
    
    # Create a heatmap for a visual representation of the crosstab
    plt.figure(figsize=(10, 8))
    sns.heatmap(generation_cluster, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Heatmap of Generation Distribution in Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Generation')
    plt.savefig('cluster_visualisations/generation_cluster_heatmap.png')
    plt.close()

# Function to create a 2D scatter plot of clusters using PCA for dimensionality reduction
def plot_2d_clusters(data, cluster_column='Cluster'):
    
    # Prepare data by selecting numeric types and removing the cluster column if present
    if cluster_column in data.columns:
        data_for_pca = data.drop(columns=[cluster_column]).select_dtypes(include=[np.number])
    else:
        data_for_pca = data.select_dtypes(include=[np.number])

    # Reduce data to 2 principal components for a 2D plot
    pca = PCA(n_components=2)
    data_transformed = pca.fit_transform(data_for_pca)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    data['Cluster'] = kmeans.fit_predict(data_transformed)

    # Plotting
    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue', 'yellow', 'purple']  # Ensure you have enough colors for the number of clusters
    num_clusters = data['Cluster'].nunique()

    for i in range(num_clusters):
        plt.scatter(data_transformed[data['Cluster'] == i, 0], data_transformed[data['Cluster'] == i, 1], s=50, c=colors[i], label=f'Cluster {i}')
    plt.title('2D Visualization of User Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig('cluster_visualisations/2d_clusters.png')
    plt.close()

# Function to create a 3D scatter plot of clusters using PCA for dimensionality reduction
def plot_3d_clusters(data, cluster_column='Cluster'):
    
    #Prepare data for PCA
    if cluster_column in data.columns:
        data_for_pca = data.drop(columns=[cluster_column]).select_dtypes(include=[np.number])
    else:
        data_for_pca = data.select_dtypes(include=[np.number])
    
    # Reduce data to 3 principal componenets for a 3D plot
    pca = PCA(n_components=3)
    data_transformed = pca.fit_transform(data_for_pca)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    for i in range(np.max(data['Cluster']) + 1):
        ax.scatter(data_transformed[data['Cluster'] == i, 0], data_transformed[data['Cluster'] == i, 1], data_transformed[data['Cluster'] == i, 2], s=50, c=colors[i], label=f'Cluster {i}')
    ax.set_title('3D Visualization of Clusters')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()
    plt.savefig('cluster_visualisations/3d_clusters.png')
    plt.close()
    
# Function to create a pair plot colored by cluster labels
def plot_pairplot(df, cluster_col):
    pair_plot = sns.pairplot(df, hue=cluster_col)
    pair_plot.savefig('cluster_visualisations/pairplot_clusters.png')
    plt.close()

