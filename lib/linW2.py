import numpy as np
import ot
from time import time
from sklearn.decomposition import PCA
import linW2_utils as LinW2


###################################################
#### Wasserstein barycenter with fixed support ####
###################################################


bary_kmeans_times = []
for nb_clusters in [16, 32, 64, 128, 256, 512] :
    kmeans_centers = np.load("../Results/KMeans/Kmeans_centers_K"+str(nb_clusters)+".npy")
    kmeans_weights = np.load("../Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy")

    t = time()
    cost_matrix = ot.dist(kmeans_centers)
    barycenter_weights = ot.lp.barycenter(np.transpose(kmeans_weights), cost_matrix)
    bary_kmeans_times.append(time()-t)
    np.save("../Results/Barycenter/barycenter_weights_Kmeans_K"+str(nb_clusters)+".npy", barycenter_weights)
np.save("../Results/Exec_times/bary_kmeans_times.npy", bary_kmeans_times)


bary_fsom_times = []
for nb_clusters in [16, 32, 64, 128, 256, 512] :
    flowsom_centers = np.load("../Results/FlowSOM/flowsom_centers_K"+str(nb_clusters)+".npy")
    flowsom_weights = np.load("../Results/FlowSOM/flowsom_weights_K"+str(nb_clusters)+".npy")

    t = time()
    cost_matrix = ot.dist(flowsom_centers)
    barycenter_weights = ot.lp.barycenter(np.transpose(flowsom_weights), cost_matrix)
    bary_fsom_times.append(time()-t)
    np.save("../Results/Barycenter/barycenter_weights_flowsom_K"+str(nb_clusters)+".npy", barycenter_weights)
np.save("../Results/Exec_times/bary_flowsom_times.npy", bary_fsom_times)


##########################################
######### Linearisation of OT ############
##########################################

lin_times = []
for nb_clusters in [16, 32, 64, 128, 256, 512] :    
    kmeans_centers = np.load("../Results/KMeans/Kmeans_centers_K"+str(nb_clusters)+".npy")
    kmeans_weights = np.load("../Results/KMeans/Kmeans_weights_K"+str(nb_clusters)+".npy")
    barycenter_support = kmeans_centers

    # Reference measure is barycenter of K-means weights, and support is K-means centers
    barycenter_weights = np.load("../Results/Barycenter/barycenter_weights_Kmeans_K"+str(nb_clusters)+".npy")
    t = time()
    lin_bary_OT = LinW2.lin_OT(barycenter_support, barycenter_weights, kmeans_centers, kmeans_weights)
    lin_times.append(time()-t)
    np.save("../Results/linW2/linW2_W2bary_Kmeans_K"+str(nb_clusters)+".npy", lin_bary_OT)

    # Reference measure has uniform weights
    unif_weights = np.array([1/nb_clusters]*nb_clusters, dtype='float')
    lin_unif_OT = LinW2.lin_OT(barycenter_support, unif_weights, kmeans_centers, kmeans_weights)
    np.save("../Results/linW2/linW2_unif_Kmeans_K"+str(nb_clusters)+".npy", lin_unif_OT)

    # Reference measure has L2 mean weights of K-means weights
    L2mean_weights = np.mean(kmeans_weights, axis=0)
    lin_L2mean_OT = LinW2.lin_OT(barycenter_support, L2mean_weights, kmeans_centers, kmeans_weights)
    np.save("../Results/linW2/linW2_L2mean_Kmeans_K"+str(nb_clusters)+".npy", lin_L2mean_OT)

    ##################
    ##### PCA ########
    ##################

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(lin_bary_OT)
    np.save("../Results/PCA/pca_linW2_W2bary_Kmeans_K"+str(nb_clusters)+".npy", data_pca)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(lin_unif_OT)
    np.save("../Results/PCA/pca_linW2_unif_Kmeans_K"+str(nb_clusters)+".npy", data_pca)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(lin_L2mean_OT)
    np.save("../Results/PCA/pca_linW2_L2mean_Kmeans_K"+str(nb_clusters)+".npy", data_pca)
np.save("../Results/Exec_times/linW2_kmeans_times.npy", np.array(lin_times))


lin_fsom_times = []
for nb_clusters in [16, 32, 64, 128, 256, 512] :    
    flowsom_centers = np.load("../Results/FlowSOM/flowsom_centers_K"+str(nb_clusters)+".npy")
    flowsom_weights = np.load("../Results/FlowSOM/flowsom_weights_K"+str(nb_clusters)+".npy")
    barycenter_support = flowsom_centers

    # Reference measure is barycenter of FlowSOM weights, and support is FlowSOM centers
    barycenter_weights = np.load("../Results/Barycenter/barycenter_weights_flowsom_K"+str(nb_clusters)+".npy")
    t = time()
    lin_bary_OT = LinW2.lin_OT(barycenter_support, barycenter_weights, flowsom_centers, flowsom_weights)
    lin_fsom_times.append(time()-t)
    np.save("../Results/linW2/linW2_W2bary_flowsom_K"+str(nb_clusters)+".npy", lin_bary_OT)

    # Reference measure has uniform weights
    unif_weights = np.array([1/nb_clusters]*nb_clusters, dtype='float')
    lin_unif_OT = LinW2.lin_OT(barycenter_support, unif_weights, flowsom_centers, flowsom_weights)
    np.save("../Results/linW2/linW2_unif_flowsom_K"+str(nb_clusters)+".npy", lin_unif_OT)

    # Reference measure has L2 mean weights of FlowSOM weights
    L2mean_weights = np.mean(flowsom_weights, axis=0)
    lin_L2mean_OT = LinW2.lin_OT(barycenter_support, L2mean_weights, flowsom_centers, flowsom_weights)
    np.save("../Results/linW2/linW2_L2mean_flowsom_K"+str(nb_clusters)+".npy", lin_L2mean_OT)

    ##################
    ##### PCA ########
    ##################

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(lin_bary_OT)
    np.save("../Results/PCA/pca_linW2_W2bary_flowsom_K"+str(nb_clusters)+".npy", data_pca)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(lin_unif_OT)
    np.save("../Results/PCA/pca_linW2_unif_flowsom_K"+str(nb_clusters)+".npy", data_pca)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(lin_L2mean_OT)
    np.save("../Results/PCA/pca_linW2_L2mean_flowsom_K"+str(nb_clusters)+".npy", data_pca)
np.save("../Results/Exec_times/linW2_flowsom_times.npy", np.array(lin_fsom_times))