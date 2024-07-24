import numpy as np
import ot

def barycentric_projection(pi, target_support) :
    '''
    Extracts a Monge map from a transport plan with the barycentric projection
    '''
    n = pi.shape[0]
    d = target_support.shape[1]
    T = np.zeros(shape=(n,d))
    for i in range(n) :
        if np.sum(pi[i])==0:
            continue
        piRow = pi[i]/np.sum(pi[i])
        T[i] = piRow@target_support
    return T

def lin_mu(mu_support, mu_weights, ref_support, ref_weights) :
    '''
    Linearisation of the discrete measure mu w.r.t a reference measure    
    '''
    cost_matrix = ot.dist(ref_support, mu_support)
    pi = ot.emd(ref_weights, mu_weights, cost_matrix)
    T = barycentric_projection(pi, mu_support)
    T = np.sqrt(ref_weights[:,np.newaxis])*(T - mu_support)
    return T

def lin_OT(ref_support, ref_weights, mu_support, mu_weights) :
    '''
    Linearisation of a set of measures on the same support (mu_support) but with different weights (mu_weights)
    w.r.t a reference measure
    '''
    lin_data = []
    N = len(mu_weights)
    for n in range(N) :
        lin_data.append(lin_mu(mu_support, mu_weights[n], ref_support, ref_weights))
    lin_data = np.array(lin_data).reshape((N,-1))
    lin_data = lin_data - np.mean(lin_data, axis=0)
    return lin_data