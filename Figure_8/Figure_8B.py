import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import sys
import matplotlib.pyplot as plt
import pyemma
from scipy import stats


a = np.load('./residue_pair_ind.npy')

indexs = np.unique(a)


###################################################################Function for K-L divergence###################################

def kl_divergence(p, q):
    kl_div = 0
    for i in range(len(p)):
        if not(p[i] == 0 or q[i] == 0):
            kl_div += p[i] * np.log2(p[i]/q[i])
    return kl_div


###################################################################Inter residue heavy atom distance loading#####################
txx_THC = np.load('partial_agonist_heavy_atom_distances.npy')

txx_AGO = np.load('agonist_heavy_atom_distances.npy')

###################################################################K-L divergence calculation####################################
number_of_feature = txx_AGO.shape[1]

kl_div = np.empty([number_of_feature,1])

for i in range(number_of_feature):
    x_data = txx_THC[:,i]
    y_data = txx_AGO[:,i]
    minimum = np.min(np.minimum(x_data,y_data))
    maximum = np.max(np.maximum(x_data,y_data))
    bins = np.arange(minimum,maximum,(maximum-minimum)/100)
    xhist,xedges = np.histogram(x_data,bins=bins,density=True)
    yhist,yedges = np.histogram(y_data,bins=bins,density=True)
    x_prob = xhist * np.diff(xedges)

    y_prob = yhist * np.diff(yedges)
    kl_div[i,0] = (kl_divergence(x_prob,y_prob) + kl_divergence(y_prob,x_prob))/2

#############################################################per residue contribution of K-L divergence####################################
weights_per_residue = np.empty([len(indexs),1])
for i in range(len(indexs)):
    position = np.where(a==indexs[i])[0]
    weights_per_residue[i,0] = np.sum(kl_div[position])



print(weights_per_residue)



