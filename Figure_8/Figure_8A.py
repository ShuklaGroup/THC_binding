import glob 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import sys
import matplotlib.pyplot as plt
import pyemma
from scipy import stats
from matplotlib import rc

#################################################################inter residue heavy atom distance###################################
txx_THC = np.load('partial_agonist_heavy_atom_distances.npy')

txx_AGO = np.load('agonist_heavy_atom_distances.npy')


#################################################################PCA implementation##################################################
pca_THC = PCA(n_components=10)

pca_THC.fit(txx_THC)


pca_AGO = PCA(n_components=10)

pca_AGO.fit(txx_AGO)

#################################################################Figure specification################################################
hfont = {'fontname':'Helvetica','fontweight':'bold'}
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

fig_wid = 10
fig_hig = 7
cmap = mpl.cm.jet

fig, axs = plt.subplots(1, 1, figsize=(10, 7))

#################################################################2-D scatter plot####################################################

axes.scatter(txx_AGO.dot(pca_AGO.components_[0]),txx_AGO.dot(pca_AGO.components_[1]),s=2,c='violet')
axes.scatter(txx_THC.dot(pca_THC.components_[0]),txx_THC.dot(pca_THC.components_[1]),s=2,c='green')

axes.get_xaxis().set_ticks([])
axes.get_yaxis().set_ticks([])

plt.xlabel('Projection on 1st PC', **hfont,fontsize=30)
plt.ylabel('Projection on 2nd PC', **hfont,fontsize=30)


plt.tight_layout()

plt.savefig('Figure_8A.png')

plt.close()


