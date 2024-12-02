from sklearn.cluster import KMeans
import numpy as np

def mean_measure_quantif(data, data_weights, nb_clusters) :
    N = len(data)
    concat_data = np.concatenate(tuple(data))
    concat_data_weights = np.concatenate(tuple(data_weights))
    kmeans = KMeans(n_clusters=nb_clusters, n_init="auto", random_state=42).fit(concat_data, sample_weight=concat_data_weights)
    kmeans_centers = kmeans.cluster_centers_
    kmeans_labels = [kmeans.predict(data[i]) for i in range(N)]

    #Computing weights as mass of Voronoi cells
    tmp_data_weights = [N*data_weights[i] for i in range(N)]
    kmeans_weights = np.array([[sum((tmp_data_weights[i])[kmeans_labels[i]==k]) for k in range(nb_clusters)] for i in range(N)])
    return kmeans_centers, kmeans_weights