import glob
import numpy as np
import os
import pyemma
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

#######################################Feature matrix loading##############################################
dica =  {'Ex13_distance':0,'Ex14_distance':1,'Ex15_distance':2,'Ex16_distance':3,'Ex17_distance':4,
         'Ex23_distance':5,'Ex24_distance':6,'Ex25_distance':7,'Ex26_distance':8,'Ex27_distance':9,
         'Helix2M_Nloop':10,'ECL2_Nloop':11,'In26_distance':12,'In36_distance':13,'In27_distance':14,
        'THC_C14_PHE':15,'THC_C16_TYR':16,
        'THC_C16_TRP_181':17,'THC_C20_TYR':18,'THC_C20_TRP_181':19,'TGPHE_chi2':20,'TGTRP_chi2':21,'THC_dihy':22
        }                                                   #MSM features defined as a dictionary

features = pickle.load(open('./../features.pkl','rb'))      #loading of trajectory feature (list of arrays)

txx = np.concatenate(features)        #concatenation of all trajectory features into a 2-D array


#######################################MSM loading##############################################
msm = pickle.load(open("./../final_MSM_obj.pkl","rb")) #loading of MSM object 

weights=np.concatenate(msm.trajectory_weights())                                           #weights(probability density of each frames)


#######################################Parameters and hyperparameters definition##############################################

x_bins = 100        #number of bins used to divide the landscape in x-direction 
y_bins = 100        #number of bins used to divide the landscape in y-direction 

R = 0.001987        #Boltzman's constant (kcal/mol/K)
T = 300             #Temperature (K)

#######################################Figure Specification##############################################
hfont = {'fontname':'Helvetica','fontweight':'bold'}
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   #Figure font definition 

fig_wid = 10        #Width of the genarated figure 
fig_hig = 7         #length of the genarated figure
cmap = mpl.cm.jet   #color bar used in the figure

Max_energy =5       #maximum energy projected in color bar 


#######################################2-D Histogram##############################################

x_key = 'TGTRP_chi2'                               #feature used here to plot in x-direction 
y_key = 'In36_distance'                             #feature used here to plot in y-direction

x_data =  txx[:,dica[x_key]]*180/np.pi             #assign x_data as 1-D array with x-direction feature (change nm to angtrom)
y_data =  txx[:,dica[y_key]]*10                    #assign y_data as 1-D array with y-direction feature (change radian to degree)

x_data_min =  np.min(x_data)                        #minimum value of x-direction feature
y_data_min =  np.min(y_data)                        #minimum value of y-direction feature
x_data_max =  np.max(x_data)                        #maximum value of x-direction feature
y_data_max =  np.max(y_data)                        #maximum value of y-direction feature

x_hist_lim_low =  x_data_min -0.5                   #mimimum limit of histogram in x-direction 
y_hist_lim_low =  y_data_min -0.5                   #mimimum limit of histogram in y-direction
x_hist_lim_high = x_data_max +0.5                   #maximum limit of histogram in x-direction
y_hist_lim_high = y_data_max  +0.5                  #maximum limit of histogram in y-direction

hist= np.histogram2d(x_data,y_data, bins=[x_bins,y_bins],
                     range = [[x_hist_lim_low,x_hist_lim_high],[y_hist_lim_low,y_hist_lim_high]],
                     density= True,weights=weights) #2-D histogram 

prob_density = hist[0]                              #probablity density obtained from histogram
xedge = hist[1]                                     #edges of the bins obtained from histogram in x-direction 
yedge = hist[2]                                     #edges of the bins obtained from histogram in y-direction 

x_bin_size = xedge[1]-xedge[0]                      #x-bin size
y_bin_size = yedge[1]-yedge[0]                      #y-bin size

#######################################Free energy calculations##############################################

free_energy = -R*T*np.log(prob_density*x_bin_size*y_bin_size)     #absolute value of the free energy in each bin 
min_free_energy= np.min(free_energy)                              #minimum value of the free energy
delta_free_energy = free_energy - min_free_energy                 #Relative free energy in each bin 


#######################################Contour plot##############################################
fig, axs = plt.subplots(1,1,figsize=(fig_wid,fig_hig))            

xx = [(xedge[i]+xedge[i+1])/2 for i in range(len(xedge)-1)]                                     #average values of x-bins
yy = [(yedge[i]+yedge[i+1])/2 for i in range(len(yedge)-1)]                                     #average values of y-bins

cd =axs.contourf(xx,yy,delta_free_energy.T,
                 np.linspace(0,Max_energy,Max_energy*5+1), vmin=0.0, vmax=Max_energy,cmap=cmap) #contour plot

cbar = fig.colorbar(cd,ticks=range(Max_energy+1))                                               #color bar
cbar.ax.set_yticklabels(range(Max_energy+1),fontsize=22)                                        #ticklabels of color bar 
cbar.ax.set_ylabel('Free Energy (Kcal/mol)', labelpad=15,**hfont,fontsize=30)                   #axis labels of color bar 

axs.set_xlim([-180,180])                                                                            #min and max limit of x-axis
axs.set_ylim([7,19])                                                                        #min and max limit of y-axis

axs.set_xticks(range(int(-180),int(180)+1,60))                                                       #x-axis ticks
axs.set_yticks(range(int(7),int(19)+1,4))                                                      #y-axis ticks

plt.xticks(fontsize=22)                                                                         #x-axis tick size
plt.yticks(fontsize=22)                                                                         #y-axis tick size

axs.set_xticklabels(range(int(-180),int(180)+1,60))                                                  #x-axis ticklabel
axs.set_yticklabels(range(int(7),int(19)+1,4))                                                 #y-axis ticklabels

plt.xlabel('Toggle Switch ' + 'TRP356$^{6.48}$ Angle' + '($\chi_2$)', **hfont,fontsize=30)                                       #x-axis label
plt.ylabel('Intracellular TM3-TM6 distance' + ' (\AA)', **hfont,fontsize=30)                                   #y-axis label

plt.tight_layout()
Filename = 'Figure_7A'
plt.savefig(Filename,transparent=True,dpi =500)

