from sklearn.metrics import silhouette_score
import numpy as np
import json

def kmeans_evaluation(data_transformed, clusters, clustered_df):
    
    # calculates the silhouette score
    silhouette = silhouette_score(data_transformed, clusters)

    # calculates the intra-cluster and inter-cluster distances
    unique_clusters = np.unique(clusters)
    intra_distances = []
    inter_distances = []
    
    for cluster in unique_clusters:
        cluster_data = data_transformed[clusters == cluster]
        intra_dist = np.mean([np.linalg.norm(cluster_data - np.mean(cluster_data, axis=0), axis=1)])
        inter_dist = np.mean([np.linalg.norm(cluster_data - np.mean(data_transformed[clusters != cluster], axis=0), axis=1)])
        intra_distances.append(intra_dist)
        inter_distances.append(inter_dist)

    # prepares data for loading into a JSON file
    metrics = {
        "Silhouette Score": silhouette,
        "Intra-Cluster Distances": intra_distances,
        "Inter-Cluster Distances": inter_distances
    }
    
    # writes data to a JSON file
    with open('cluster_evaluations/kmeans_metrics.json', 'w') as file:
        json.dump(metrics, file)
