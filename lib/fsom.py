import flowsom as fs
import numpy as np
import pandas as pd
from time import time

hipc_df = pd.read_csv("../Data/hipc_df.csv")
N = len(hipc_df)
file_names = hipc_df["File"].to_numpy()


ff = fs.io.read_FCS("../Data/merged_data.fcs")

xy =[[4,4],[4,8],[8,8],[8,16],[16,16],[16,32]]
nb_clusters = [16,32,64,128,256,512]
exec_times = []
for index,K in enumerate(nb_clusters) :
    t = time()
    fsom = fs.FlowSOM(ff, cols_to_use=[0,1,2,3,4,5,6], xdim=xy[index][0], ydim=xy[index][1], n_clusters=10, seed=42)
    exec_times.append(time()-t)

    flowsom_centers = fsom.model.codes
    flowsom_labels = fsom.model.cluster_labels

    data = [np.asarray(pd.read_csv("../Data/CSV/"+file_names[i], usecols = np.arange(1,8))) for i in range(N)]
    data_lengths = [len(data[i]) for i in range(N)]

    j = 0
    flowsom_weights = []
    for i in range(N) : 
        data_i_labels = flowsom_labels[j:j+data_lengths[i]]
        j += data_lengths[i]
        flowsom_weights.append(np.array([len(data_i_labels[data_i_labels==k])/len(data_i_labels) 
                                for k in range(K)]))
        
    flowsom_weights = np.array(flowsom_weights)
    np.save("../Results/FlowSOM/flowsom_centers_K"+str(K)+".npy", flowsom_centers)
    np.save("../Results/FlowSOM/flowsom_weights_K"+str(K)+".npy", flowsom_weights)

np.save("../Results/Exec_times/flowsom_times.npy", exec_times)
