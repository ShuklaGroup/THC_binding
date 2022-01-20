import mdtraj as md
import glob
import numpy as np
import os
import pyemma
import math
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import MeanShift
import random
from matplotlib import rc

########################################################Figure Specification###############################################

hfont = {'fontname':'Helvetica','fontweight':'bold'}
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

fig_wid = 10
fig_hig = 7
cmap = mpl.cm.jet

fig, axs = plt.subplots(1, 1, figsize=(10, 7))


######################################################activation matrix loading###########################################

tpa = np.load('partial_agonist_toggle_switch_TM6_movement.npy')

ta = np.load('agonist_toggle_switch_TM6_movement.npy')

#####################################################2-D scatter plot#####################################################

axs.scatter(ta[:,0]*10,ta[:,1]*10,s=2,c='violet',label="Agonist")

axs.scatter(tpa[:,0]*10,tpa[:,1]*10,s=2,c='green',label="Partial Agonist")


inactive = [7.9054385,11.950634]           #inactive crystal structure 
active_like = [10.914677,15.470214]        #active crystal structure 

axs.scatter(inactive[0],inactive[1],s=100,c='black',marker='^')
axs.scatter(active_like[0],active_like[1],s=100,c='black',marker='o')

axs.set_xlim([6,14])
axs.set_ylim([8,20])


axs.set_xticks(range(int(6),int(14)+1,2))
axs.set_xticklabels(range(int(6),int(14)+1,2))

axs.set_yticks(range(int(8),int(20)+1,2))
axs.set_yticklabels(range(int(8),int(20)+1,2))



plt.xlabel('TRP356$^{6.48}$-ASP163$^{2.50}$ distance' + ' (\AA)', **hfont,fontsize=30)
plt.ylabel('ARG214$^{3.50}$-LYS343$^{6.35}$ distance (\AA)', **hfont,fontsize=30)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.tight_layout()

plt.savefig('Figure_8C',transparent=False,dpi =500)

