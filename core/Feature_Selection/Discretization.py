import numpy as np


#Bin discretization
def bins_discretizer(x, frac = 0.1):
    b = int(np.unique(x).shape[0]*frac)
    bins = np.linspace(x.min(), x.max(), b)
    x_cat = np.digitize(x, bins)
    return x_cat