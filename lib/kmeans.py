import numpy as np
import pandas as pd
import time

from sklearn import preprocessing
from sklearn.cluster import KMeans

hipc_df = pd.read_csv("../Data/hipc_df.csv")
N = len(hipc_df)
file_names = hipc_df["File"].to_numpy()

data = [np.asarray(pd.read_csv("../Data/CSV/"+file_names[i], usecols = np.arange(1,8))) for i in range(N)]
data_weights = [np.array([1/(N*len(data[i]))]*len(data[i]), dtype='float') for i in range(N)]

data_normed = []
for i in range(N) :
    data_normed.append(preprocessing.StandardScaler().fit_transform(data[i]))

concat_data = np.concatenate(tuple(data_normed))
concat_data_weights = np.concatenate(tuple(data_weights))

exec_times = []
for nb_clusters in [16,32,64,128,256,512] :
    t = time.time()
    kmeans = KMeans(n_clusters=nb_clusters, n_init="auto", random_state=42).fit(concat_data, sample_weight=concat_data_weights)
    kmeans_centers = kmeans.cluster_centers_

    kmeans_labels = [kmeans.predict(data_normed[i]) for i in range(N)]

    #Computing weights as mass of Voronoi cells
    kmeans_weights = np.array([[len(kmeans_labels[i][kmeans_labels[i]==k])/len(kmeans_labels[i]) 
                            for k in range(nb_clusters)]
                            for i in range(N)], dtype='float') 
    exec_times.append(time.time()-t)

    np.save("../Results/KMeans/Kmeans_centers_K"+str(nb_clusters)+".npy", kmeans_centers)
    np.save("../Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy", kmeans_weights)

np.save("../Results/Exec_times/KMeans_times", np.array(exec_times))
