import numpy as np
import pandas as pd
from time import time

def estimate_sigma(data) :
    ''' Estimates the bandwidth to take for the RBF kernel as the median of the squared distance
    between all pairs of points

    
    This could be very long, so we do not cover all data points
    '''
    sigma_list = []
    for i in range(0,len(data), len(data)) :
        #Compute the median within a dataset
        tmp_sigma = np.median([np.linalg.norm(data[i][i1]-data[i][i2]) for i1 in range(0,len(data[i]),50) for i2 in range(i1+1,len(data[i]),50)])
        sigma_list.append(tmp_sigma)

    #Compute the median of medians
    return np.median(np.array(sigma_list))



def featureMapping(x, sigma, s) :
    ''' Random Fourier Feature mapping with RBF kernel'''
    d = len(x)
    cov = np.diag(np.array([sigma**(-2)]*d, dtype='float'))
    W = np.random.multivariate_normal(np.zeros(d), cov, s//2)
    return np.concatenate((np.sin(W@x), np.cos(W@x)))



def measureEmbedding(P, sigma, s) :
    ''' Embedding of the measure P via the means of all features mapping '''
    cpt = 0
    for i in range(len(P)) :
        cpt += featureMapping(P[i], sigma, s)
    return cpt/len(P)
