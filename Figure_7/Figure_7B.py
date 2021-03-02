import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from matplotlib import rc

#######################################Initialization##############################################
dica =  {'Ex13_distance':0,'Ex14_distance':1,'Ex15_distance':2,'Ex16_distance':3,'Ex17_distance':4,
         'Ex23_distance':5,'Ex24_distance':6,'Ex25_distance':7,'Ex26_distance':8,'Ex27_distance':9,
         'Helix2M_Nloop':10,'ECL2_Nloop':11,'In26_distance':12,'In36_distance':13,'In27_distance':14,
        'THC_C14_PHE':15,'THC_C16_TYR':16,
        'THC_C16_TRP_181':17,'THC_C20_TYR':18,'THC_C20_TRP_181':19,'TGPHE_chi2':20,'TGTRP_chi2':21,'THC_dihy':22
        }                                                   #MSM features defined as a dictionary


#Initialization of 2-D array to contain Conditional probability for each bootstrap sample with TRP356 in inactive and partially active position 
fraction_inactive = np.empty([200,2])                       #First and Second columns contain the C.P. of TM6 in in extended or inactive 
fraction_pa = np.empty([200,2])                       #position respectively


#######################################Figure Specification##############################################
hfont = {'fontname':'Helvetica'}
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) #Figure font definition

fig_wid = 10        #Width of the genarated figure
fig_hig = 7         #length of the genarated figure

def setBoxColors(bp):
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['fliers'][0], color='blue')
    plt.setp(bp['medians'][0], color='blue')

    plt.setp(bp['boxes'][1], color='red')
    plt.setp(bp['caps'][2], color='red')
    plt.setp(bp['caps'][3], color='red')
    plt.setp(bp['whiskers'][2], color='red')
    plt.setp(bp['whiskers'][3], color='red')
    plt.setp(bp['fliers'][1], color='red')
    plt.setp(bp['medians'][1], color='red')


#######################################Conditional probability for  bootstrap samples##############################################
for j in range(200):
    msm = pickle.load(open("./../bootstraping/bt_80_" + str(j) +"_msm.pkl",'rb'))
    weights = np.concatenate(msm.trajectory_weights())

    features = pickle.load(open("./../bootstraping/bt_80_" + str(j) +'_features.pkl' ,'rb'))     #loading features from bootstrap samples
    txx = np.concatenate(features)                                                          #concatenation of bootstrapped trajectory feature
    TGTRP_chi2 = txx[:,dica['TGTRP_chi2']]*180/np.pi
    In36_distance = txx[:,dica['In36_distance']]*10

    #P(TM3-TM6 distance > 12.5 A | TRP356 is in inactive state)
    bin_TGTRP_chi2 = np.where(TGTRP_chi2>60,1,0)
    bin_In36_distance = np.where(In36_distance>12.5,1,0)
    
    condition_1 = bin_TGTRP_chi2
    condition_2 = np.multiply(condition_1,bin_In36_distance)

    fraction_inactive[j,0] = np.dot(condition_2,weights)/np.dot(condition_1,weights)

    #P(TM3-TM6 distance < 12.5 A | TRP356 is in inactive state)
    bin_In36_distance = np.where(In36_distance<12.5,1,0)
    
    condition_2 = np.multiply(condition_1,bin_In36_distance)

    fraction_inactive[j,1] = np.dot(condition_2,weights)/np.dot(condition_1,weights)
    
    #P(TM3-TM6 distance > 12.5 A | TRP356 is in partially active state)
    bin_TGTRP_chi2 = np.where(TGTRP_chi2<40,1,0)
    bin_In36_distance = np.where(In36_distance>12.5,1,0)
    
    condition_1 = bin_TGTRP_chi2
    condition_2 = np.multiply(condition_1,bin_In36_distance)

    fraction_pa[j,0] = np.dot(condition_2,weights)/np.dot(condition_1,weights)

    #P(TM3-TM6 distance < 12.5 A | TRP356 is in partially active state)
    bin_In36_distance = np.where(In36_distance<12.5,1,0)
    
    condition_1 = bin_TGTRP_chi2
    condition_2 = np.multiply(condition_1,bin_In36_distance)

    fraction_pa[j,1] = np.dot(condition_2,weights)/np.dot(condition_1,weights)

print('P(TM3-TM6 distance > 12.5 A | TRP356 is in inactive state) : ')
print(np.mean(fraction_inactive[:,0]),np.std(fraction_inactive[:,0]))
print('P(TM3-TM6 distance < 12.5 A | TRP356 is in inactive state) : ')
print(np.mean(fraction_inactive[:,1]),np.std(fraction_inactive[:,1]))
print('P(TM3-TM6 distance > 12.5 A | TRP356 is in partially active state) : ')
print(np.mean(fraction_pa[:,0]),np.std(fraction_pa[:,0]))
print('P(TM3-TM6 distance < 12.5 A | TRP356 is in partially active state) : ')
print(np.mean(fraction_pa[:,1]),np.std(fraction_pa[:,1]))


#######################################ploting##############################################
fig, axs = plt.subplots(1,1,figsize=(fig_wid,fig_hig))

bp = axs.boxplot(fraction_inactive, positions = [1, 2], widths = 0.6)
setBoxColors(bp)
bp = axs.boxplot(fraction_pa, positions = [4, 5], widths = 0.6)
setBoxColors(bp)

plt.xlim(0,6)         #min and max limit of x-axis
plt.ylim(0,1)         #min and max limit of y-axis


axs.set_xticklabels(['THC Sidechain (+)', 'THC Sidechain (-)'])
axs.set_xticks([1.5, 4.5])
plt.xticks(fontsize=30)
plt.yticks(fontsize=22)

plt.ylabel('Conditional Probability',**hfont,fontsize=30)

Filename = 'Figure_7B'
plt.savefig(Filename, dpi=500)
