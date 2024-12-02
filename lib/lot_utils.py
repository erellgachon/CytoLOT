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

def log(mu_support, mu_weights, ref_support, ref_weights) :
    '''
    Linearisation of the discrete measure mu w.r.t a reference measure    
    '''
    mu_weights[mu_weights<10**(-10)] = 0
    ref_weights[ref_weights<10**(-10)] = 0

    #Computing an optimal transport plan between ref and mu
    cost_matrix = ot.dist(ref_support, mu_support)
    pi = ot.emd(ref_weights, mu_weights, cost_matrix)

    #Computing an transport map from an optimal transport plan
    T = barycentric_projection(pi, mu_support)

    #The logarithmic map maps mu to T- id
    T = T - ref_support
    return T

def LOT(ref_support, ref_weights, mu_support, mu_weights) :
    '''
    Linearisation of a set of measures with supports (mu_supports) and with weights (mu_weights)
    w.r.t a reference measure
    '''
    lin_data = []
    N = len(mu_weights)
    for n in range(N) :
        lin_data.append(log(mu_support[n], mu_weights[n], ref_support, ref_weights))
    lin_data = np.array(lin_data)
    return lin_data