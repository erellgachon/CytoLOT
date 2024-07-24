import numpy as np
import pandas as pd 

from sklearn.decomposition import PCA

def centered_log_ratio_transform(compositional):
    """Applies the centered log-ratio transform to compositional data.
    Copied from https://github.com/lucapton/crowd_labeling/tree/master    
    """

    continuous = np.log(compositional + np.finfo(compositional.dtype).eps)
    continuous -= continuous.mean(-1, keepdims=True)
    return continuous

for nb_clusters in [16, 32, 64, 128, 256, 512] :
    print(nb_clusters)
    kmeans_weights = np.load("../Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy")
    lR = centered_log_ratio_transform(kmeans_weights)
    np.save("../Results/Compositional/lr_Kmeans_K"+str(nb_clusters)+".npy", lR)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(lR)
    np.save("../Results/PCA/pca_weights_lr_Kmeans_K"+str(nb_clusters)+".npy", data_pca)