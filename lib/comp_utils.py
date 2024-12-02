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
