import numpy as np
import pandas as pd
from time import time

from sklearn.decomposition import PCA
from sklearn import preprocessing

import kme_utils as KME

hipc_df = pd.read_csv("../Data/hipc_df.csv")
N = len(hipc_df)
labels = hipc_df['Label'].to_numpy()
patient_labels =  hipc_df['Patient'].to_numpy()
labs_labels = hipc_df["Lab"].to_numpy()
file_names = hipc_df["File"].to_numpy()

data = [np.asarray(pd.read_csv("../Data/CSV/"+file_names[i], usecols = np.arange(1,8))) for i in range(N)]
data = [preprocessing.StandardScaler().fit_transform(data[i]) for i in range(N)]
data_weights = [np.array([1/(N*len(data[i]))]*len(data[i]), dtype='float') for i in range(N)]

sigma = np.round(KME.estimate_sigma(data),2)

s = 64*7 # K*d = 64*7
exec_times = []
for sigma in [0.01, 0.1, 1, 5, 10, 50, sigma] :
    t = time()
    embedded_data = np.array([KME.measureEmbedding(data[i], sigma, s) for i in range(N)], dtype="float")
    exec_times.append(time()-t)
    np.save("../Results/KME/KME_s"+str(s)+"_sigma"+str(sigma)+".npy", embedded_data)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(embedded_data)
    np.save("../Results/PCA/pca_kme_s"+str(s)+"_sigma"+str(sigma)+".npy", data_pca)

np.save("../Results/Exec_times/KME_times.npy", np.array(exec_times))