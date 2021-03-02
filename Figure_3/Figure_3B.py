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
msm_ref = pickle.load(open("./../final_MSM_obj.pkl","rb")) #loading of MSM object 

weights_ref = np.concatenate(msm_ref.trajectory_weights()) #weights(probability density of each frames)

#######################################Parameters and hyperparameters definition##############################################

bins = 100        #number of bins used to divide the landscape in x-direction

R = 0.001987        #Boltzman's constant (kcal/mol/K)
T = 300             #Temperature (K)

#######################################Figure Specification##############################################
hfont = {'fontname':'Helvetica','fontweight':'bold'}
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   #Figure font definition

fig_wid = 10        #Width of the genarated figure
fig_hig = 7         #length of the genarated figure


#######################################Ref histogram Calculation##############################################
x_key = 'THC_C16_TYR'                                           #feature used here to plot in x-direction 

x_data =  txx[:,dica[x_key]]*10                                 #assign x_data as 1-D array with x-direction feature (change nm to angtrom)

x_data_min =  np.min(x_data)                                    #minimum value of x-direction feature
x_data_max =  np.max(x_data)                                    #maximum value of x-direction feature

x_hist_lim_low =  x_data_min -0.5                               #mimimum limit of histogram in x-direction 
x_hist_lim_high =  x_data_max +0.5                              #maximum limit of histogram in x-direction

nSD, binsSD= np.histogram(x_data, bins=bins,                    #1-D histogram 
                     range = (x_hist_lim_low,x_hist_lim_high),
                     density= True,weights=weights_ref)


#######################################Ref free energy Calculation##############################################
prob_density_ref = nSD                                #probablity density obtained from histogram
area_ref = abs(binsSD[1] - binsSD[0])                 #x-bin size 

free_energy_ref = -R*T*np.log(prob_density_ref*area_ref) #absolute value of the free energy in each bin
min_free_energy_ref = np.min(free_energy_ref)            #minimum value of the free energy
delta_free_en_ref = free_energy_ref - min_free_energy_ref#Relative free energy in each bin


#######################################Energy Calculation from bootstrap samples##############################################
fr_en = np.empty([200,bins])                                                        #Initialization of 2-D array to contain free energy values
                                                                                    # from 200 bootstrap samples
count = 0
for file in glob.glob("./../bootstraping/bt_80_*_msm.pkl"):
    msm = pickle.load(open(file,'rb'))                                              #loading msm for bootstrap samples
    weights=np.concatenate(msm.trajectory_weights())                                
    
    features = pickle.load(open(file.split('msm.pkl')[0]+'features.pkl' ,'rb'))     #loading features from bootstrap samples 
    txx = np.concatenate(features)                                                  #concatenation of bootstrapped trajectory feature
    x_data = txx[:,dica[x_key]]*10                                                  #x_data is assigned with required features

    nSD, binsSD= np.histogram(x_data, bins, range = (x_hist_lim_low,x_hist_lim_high)#histogram calculation  
                ,density=True, weights=weights)
    del weights                                                                     #delete unrequired variables
    del msm                                                                         #delete unrequired variables
    del txx                                                                         #delete unrequired variables
    del features                                                                    #delete unrequired variables 
    
    prob_density = nSD                                                              #probablity density obtained from histogram
    area = abs(binsSD[1] - binsSD[0])                                               #x-bin size

    free_energy = -R*T*np.log(prob_density*area)                                    #absolute value of the free energy in each bin
    min_free_energy= np.min(free_energy)                                            #minimum value of the free energy
    delta_free_energy = free_energy - min_free_energy                               ##Relative free energy in each bin

    fr_en[count,:] = delta_free_energy                                              #loading relative free energy values for each bootstrap 
    count = count + 1


#######################################Mean and Std. Dev. calculation from bootstrap##############################################
averageSD = []                                          #Empty list for capturing the mid-points of bins 
mean_free_energy = []                                   #Empty list for capturing mean free energy for bins from bootstrap traj
err_free_energy = []                                    #Empty list for capturing error free energy for bins from bootstrap traj
ref_free_energy = []                                    #Empty list for capturing refence free energy for bins from original data

for i in range(bins):
    temp = fr_en[:,i]                                   
    if np.inf not in temp:                              #ignoring bins where one of bootstrapped msm has no data
        err_free_energy.append(np.std(temp))
        mean_free_energy.append(np.mean(temp))
        ref_free_energy.append(delta_free_en_ref[i])
        averageSD.append((binsSD[i]+binsSD[i+1])/2)

err_free_energy = np.array(err_free_energy)             #list to array
mean_free_energy = np.array(mean_free_energy)           #list to array
ref_free_energy = np.array(ref_free_energy)             #list to array

#######################################ploting##############################################
fig,axs = plt.subplots(1,1,figsize=(fig_wid,fig_hig),constrained_layout=True)

plt.plot(averageSD,ref_free_energy,color='blue')                              #Ref energy plot              
plt.fill_between(averageSD, mean_free_energy+err_free_energy, 
                mean_free_energy-err_free_energy, color='blue', alpha=0.3)    #Error bar plot

axs.set_xlim([9,21])                                                          #min and max limit of x-axis
axs.set_ylim([0,5])                                                           #min and max limit of y-axis

axs.set_xticks(range(int(9),int(21)+1,4))                                     #x-axis ticks
axs.set_yticks(range(int(0),int(5)+1,1))                                      #y-axis ticks

plt.xticks(fontsize=22)                                                       #x-axis tick font size
plt.yticks(fontsize=22)                                                       #y-axis tick font size

axs.set_xticklabels(range(int(9),int(21)+1,4))                                #x-axis ticklabels
axs.set_yticklabels(range(int(0),int(5)+1,1))                                 #y-axis ticklabels

plt.xlabel('THC Binding' + ' (\AA)', **hfont,fontsize=30)                     #x-axis label
plt.ylabel('Free Energy (Kcal/mol)',**hfont,fontsize=30)                      #y-axis label

Filename = "Figure_3B"
plt.savefig(Filename,transparent=True,dpi =500)

