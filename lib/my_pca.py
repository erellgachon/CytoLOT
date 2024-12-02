import numpy as np

def k(kernel, x, y, sigma=1) :
    if kernel=="rbf" :
        return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))
    elif kernel =="dot_product" :
        return np.dot(x,y)
    elif kernel == "quadratic_dot_product" :
        return np.dot(x,y)+np.dot(x,y)**2

def L2_k(ref_weights, T1, T2) :
     '''
     Computed inner product in L2(rho) where rho has weights ref_weights
     '''
     return np.sum([ref_weights[i]*np.dot(T1[i],T2[i]) for i in range(len(ref_weights))])

def pca_from_Gram(G, n_components=2) :
    """
    Performs PCA from the matrix of pairwise inner-products
    
    """
    # Center the kernel matrix
    n_samples = G.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples
    G_centered = G - one_n @ G - G @ one_n + one_n @ G @ one_n

    # Eigen decomposition of the centered kernel matrix
    eigenvalues, eigenvectors = np.linalg.eigh(G_centered)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1)
    
    # Select the top n_components
    alphas = eigenvectors[:, : n_components]
    lambdas = eigenvalues[: n_components]

    # Transform the data
    X_transformed = alphas * np.sqrt(lambdas)

    return X_transformed



