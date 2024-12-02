import numpy as np
import pandas as pd
from time import time
import ot
import flowsom as fs
from sklearn import preprocessing

import sys
import os
sys.path.append('../../lib')
from comp_utils import centered_log_ratio_transform
from quantif_utils import mean_measure_quantif
from lot_utils import LOT
from my_pca import pca_from_Gram, L2_k
from kme_utils import measureEmbeddingRFF, estimate_sigma

hipc_df = pd.read_csv("./Data/hipc_df.csv")
N = len(hipc_df)
file_names = hipc_df["File"].to_numpy()

list_nb_clusters = np.arange(10,250,10)
#list_nb_clusters = [10]

print("Loading data...")
data = [np.asarray(pd.read_csv("./Data/CSV/"+file_names[i], usecols = np.arange(1,8))) for i in range(N)]
data_weights = [np.array([1/(N*len(data[i]))]*len(data[i]), dtype='float') for i in range(N)]
data_lengths = [len(data[i]) for i in range(N)]
print("Data successfully loaded ! Ready for K-means")

data_normed = []
for i in range(N) :
    data_normed.append(preprocessing.StandardScaler().fit_transform(data[i]))

print("Starting KMeans for K=")
exec_times = []
for nb_clusters in list_nb_clusters :
    print(nb_clusters)
    t = time()
    centers, weights = mean_measure_quantif(data_normed, data_weights, nb_clusters)
    exec_times.append(time()-t)

    np.save("./Results/KMeans/Kmeans_centers_K"+str(nb_clusters)+".npy", centers)
    np.save("./Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy", weights)
np.save("./Results/Exec_times/KMeans_times", np.array(exec_times))
print("KMeans successful !")


ff = fs.io.read_FCS("./Data/merged_data.fcs")

print("Starting FlowSOM...")
exec_times = []
for index,K in enumerate(list_nb_clusters) :
    print("K =", K)
    t = time()
    fsom = fs.FlowSOM(ff, cols_to_use=[0,1,2,3,4,5,6], xdim=K//10, ydim=10, n_clusters=K, seed=42)
    exec_times.append(time()-t)

    flowsom_centers = fsom.model.codes
    flowsom_labels = fsom.model.cluster_labels

    j = 0
    flowsom_weights = []
    for i in range(N) : 
        data_i_labels = flowsom_labels[j:j+data_lengths[i]]
        j += data_lengths[i]
        flowsom_weights.append(np.array([len(data_i_labels[data_i_labels==k])/len(data_i_labels) 
                                for k in range(K)]))
        
    flowsom_weights = np.array(flowsom_weights)
    print("shape of flowsom weights :", flowsom_weights.shape)
    np.save("./Results/FlowSOM/flowsom_centers_K"+str(K)+".npy", flowsom_centers)
    np.save("./Results/FlowSOM/flowsom_weights_K"+str(K)+".npy", flowsom_weights)

np.save("./Results/Exec_times/flowsom_times.npy", exec_times)


###################################################
########## Compositional data analysis ############
###################################################
print("Starting compositional data analysis...")
exec_times_LR = []
for nb_clusters in list_nb_clusters :
    print(nb_clusters)
    kmeans_weights = np.load("./Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy")
    t = time()
    lR = centered_log_ratio_transform(kmeans_weights)        

    K_matrix = np.array([[np.dot(lR[i], lR[j]) for i in range(N)] for j in range(N)])
    comp_data_pca = pca_from_Gram(K_matrix, 3)
    comp_data_pca_normed = comp_data_pca / np.linalg.norm(comp_data_pca, axis= 0)
    exec_times_LR.append(time()-t)
    np.save("./Results/Compositional/lr_Kmeans_K"+str(nb_clusters)+".npy", lR)
    np.save("./Results/PCA/pca_weights_lr_Kmeans_K"+str(nb_clusters)+".npy", comp_data_pca_normed)
np.save("./Results/Exec_times/comp_pca_times.npy", np.array(exec_times_LR))

print("Compositional data analysis successful !")


##########################################
######### Linearisation of OT ############
##########################################

print("Starting linearisation for KMeans...")
print("Linearisation with ref = barycenter")
lin_pca_times = []
for nb_clusters in list_nb_clusters :    
    print("K = ", nb_clusters)
    kmeans_centers = np.load("./Results/KMeans/Kmeans_centers_K"+str(nb_clusters)+".npy")
    kmeans_weights = np.load("./Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy")
    t = time()
    
    cost_matrix = ot.dist(kmeans_centers)
    barycenter_weights = ot.lp.barycenter(np.transpose(kmeans_weights), cost_matrix)
    ref_support = kmeans_centers
    ref_weights = barycenter_weights

    # Reference measure is barycenter of K-means weights, and support is K-means centers    ref_weights = np.load("./Results/Barycenter/barycenter_weights_Kmeans_K"+str(nb_clusters)+".npy")
    lin_OT = LOT(ref_support, ref_weights, [kmeans_centers for i in range(len(kmeans_weights))], kmeans_weights)
    
    # PCA
    K_matrix = np.array([[L2_k(ref_weights, lin_OT[i], lin_OT[j]) for i in range(N)] for j in range(N)])
    lot_data_pca = pca_from_Gram(K_matrix, 3)
    lot_data_pca_normed = lot_data_pca / np.linalg.norm(lot_data_pca, axis= 0)
    lin_pca_times.append(time()-t)

    np.save("./Results/Barycenter/barycenter_weights_Kmeans_K"+str(nb_clusters)+".npy", barycenter_weights)
    np.save("./Results/linW2/linW2_W2bary_Kmeans_K"+str(nb_clusters)+".npy", lin_OT)
    np.save("./Results/PCA/pca_linW2_W2bary_Kmeans_K"+str(nb_clusters)+".npy", lot_data_pca_normed)
np.save("./Results/Exec_times/linW2_W2bary_kmeans_pca_times.npy", np.array(lin_pca_times))

print("Linearisation with ref=L2mean")
lin_pca_times = []
for nb_clusters in list_nb_clusters :    
    print("K = ", nb_clusters)
    kmeans_centers = np.load("./Results/KMeans/Kmeans_centers_K"+str(nb_clusters)+".npy")
    kmeans_weights = np.load("./Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy")
    N = len(kmeans_weights)
    ref_support = kmeans_centers

    # Reference measure is uniform of size K-means weights, and support is K-means centers
    ref_weights = np.mean(kmeans_weights, axis=0)
    t = time()
    lin_OT = LOT(ref_support, ref_weights, [kmeans_centers for i in range(len(kmeans_weights))], kmeans_weights)
    
    # PCA
    K_matrix = np.array([[L2_k(ref_weights, lin_OT[i], lin_OT[j]) for i in range(N)] for j in range(N)])
    lot_data_pca = pca_from_Gram(K_matrix, 3)
    lot_data_pca_normed = lot_data_pca / np.linalg.norm(lot_data_pca, axis= 0)
    lin_pca_times.append(time()-t)
    
    np.save("./Results/linW2/linW2_L2mean_Kmeans_K"+str(nb_clusters)+".npy", lin_OT)
    np.save("./Results/PCA/pca_linW2_L2mean_Kmeans_K"+str(nb_clusters)+".npy", lot_data_pca_normed)
np.save("./Results/Exec_times/linW2_L2mean_kmeans_pca_times.npy", np.array(lin_pca_times))

print("Linearisation with ref=uniform")
lin_pca_times = []
for nb_clusters in list_nb_clusters :    
    print("K = ", nb_clusters)
    kmeans_centers = np.load("./Results/KMeans/Kmeans_centers_K"+str(nb_clusters)+".npy")
    kmeans_weights = np.load("./Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy")
    N = len(kmeans_weights)
    ref_support = kmeans_centers

    # Reference measure is uniform of size K-means weights, and support is K-means centers
    ref_weights = np.array([1/nb_clusters]*nb_clusters, dtype='float')
    t = time()
    lin_OT = LOT(ref_support, ref_weights, [kmeans_centers for i in range(len(kmeans_weights))], kmeans_weights)
    
    # PCA
    K_matrix = np.array([[L2_k(ref_weights, lin_OT[i], lin_OT[j]) for i in range(N)] for j in range(N)])
    lot_data_pca = pca_from_Gram(K_matrix, 3)
    lot_data_pca_normed = lot_data_pca / np.linalg.norm(lot_data_pca, axis= 0)
    lin_pca_times.append(time()-t)

    np.save("./Results/linW2/linW2_unif_Kmeans_K"+str(nb_clusters)+".npy", lin_OT)
    np.save("./Results/PCA/pca_linW2_unif_Kmeans_K"+str(nb_clusters)+".npy", lot_data_pca_normed)
np.save("./Results/Exec_times/linW2_unif_kmeans_pca_times.npy", np.array(lin_pca_times))

print("Linearisation with ref=random choice among measures")
lin_pca_times = []
for nb_clusters in list_nb_clusters :    
    print("K = ", nb_clusters)
    kmeans_centers = np.load("./Results/KMeans/Kmeans_centers_K"+str(nb_clusters)+".npy")
    kmeans_weights = np.load("./Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy")
    N = len(kmeans_weights)
    ref_support = kmeans_centers

    # Reference measure is uniform of size K-means weights, and support is K-means centers
    rand_index = np.random.choice(N)
    ref_weights = kmeans_weights[rand_index]
    t = time()
    lin_OT = LOT(ref_support, ref_weights, [kmeans_centers for i in range(len(kmeans_weights))], kmeans_weights)
    
    # PCA
    K_matrix = np.array([[L2_k(ref_weights, lin_OT[i], lin_OT[j]) for i in range(N)] for j in range(N)])
    lot_data_pca = pca_from_Gram(K_matrix, 3)
    lot_data_pca_normed = lot_data_pca / np.linalg.norm(lot_data_pca, axis= 0)
    lin_pca_times.append(time()-t)

    np.save("./Results/linW2/linW2_rand_Kmeans_K"+str(nb_clusters)+".npy", lin_OT)
    np.save("./Results/PCA/pca_linW2_rand_Kmeans_K"+str(nb_clusters)+".npy", lot_data_pca_normed)
np.save("./Results/Exec_times/linW2_rand_kmeans_pca_times.npy", np.array(lin_pca_times))



print("Starting linearisation for FlowSOM...")
print("Linearisation with ref = barycenter")
lin_pca_times = []
for nb_clusters in list_nb_clusters :    
    print("K = ", nb_clusters)
    centers = np.load("./Results/FlowSOM/flowsom_centers_K"+str(nb_clusters)+".npy")
    weights = np.load("./Results/FlowSOM/flowsom_weights_K"+str(nb_clusters)+".npy")
    t = time()
    
    cost_matrix = ot.dist(centers)
    barycenter_weights = ot.lp.barycenter(np.transpose(weights), cost_matrix)
    ref_support = centers
    ref_weights = barycenter_weights

    # Reference measure is barycenter of K-means weights, and support is K-means centers
    lin_OT = LOT(ref_support, ref_weights, [centers for i in range(len(weights))], weights)
    
    # PCA
    K_matrix = np.array([[L2_k(ref_weights, lin_OT[i], lin_OT[j]) for i in range(N)] for j in range(N)])
    lot_data_pca = pca_from_Gram(K_matrix, 3)
    lot_data_pca_normed = lot_data_pca / np.linalg.norm(lot_data_pca, axis= 0)
    lin_pca_times.append(time()-t)

    np.save("./Results/Barycenter/barycenter_weights_flowsom_K"+str(nb_clusters)+".npy", barycenter_weights)
    np.save("./Results/linW2/linW2_W2bary_flowsom_K"+str(nb_clusters)+".npy", lin_OT)
    np.save("./Results/PCA/pca_linW2_W2bary_flowsom_K"+str(nb_clusters)+".npy", lot_data_pca_normed)
np.save("./Results/Exec_times/linW2_W2bary_flowsom_pca_times.npy", np.array(lin_pca_times))



#########################################
################### KME #################
#########################################

data_weights = [np.array([1/(N*len(data[i]))]*len(data[i]), dtype='float') for i in range(N)]

#sigma = np.round(estimate_sigma(data_normed),2)
#sigma=1
s = 64*7 # K*d = 64*7
exec_times = []
#list_sigma = [0.01, 0.1, 1, 5, 10]
list_sigma = [1]
print("Starting KME...")
for sigma in list_sigma :
    t = time()
    embedded_data = np.array([measureEmbeddingRFF(data_normed[i], sigma, s) for i in range(N)], dtype="float")
    exec_times.append(time()-t)
    np.save("./Results/KME/KME_s"+str(s)+"_sigma"+str(sigma)+".npy", embedded_data)

    K_matrix = np.array([[np.dot(embedded_data[i], embedded_data[j]) for i in range(N)] for j in range(N)])
    kme_data_pca = pca_from_Gram(K_matrix, 2)
    kme_data_pca_normed = kme_data_pca / np.linalg.norm(kme_data_pca, axis= 0)
    np.save("./Results/PCA/pca_kme_s"+str(s)+"_sigma"+str(sigma)+".npy", kme_data_pca_normed)

np.save("./Results/Exec_times/KME_times.npy", np.array(exec_times))
print("KME finished !")

