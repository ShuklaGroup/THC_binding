import numpy as np
import glob
import pickle
import pyemma

#######################################Initialization##############################################
#Macrostates (consist of microstates) definition 
inactive = [2, 165, 79, 28, 132]
paI = [107, 115, 39, 128, 15]
paII = [81, 170, 108, 90, 52]

#Initialization of 2-D array
MFPT_inactive_paI = np.empty([200,2])
MFPT_paI_paII = np.empty([200,2])

#####################################TPT calculations for each bootstrap sample####################
for j in range(200):
    
    msm = pickle.load(open("./../bootstraping/bt_80_" + str(j) +"_msm.pkl",'rb'))    
    #indexes of the macrostates in the transition probability matrix of bootstrap sample
    inactive_prime = [msm.active_set.tolist().index(s) for s in inactive if s in msm.active_set]
    paI_prime = [msm.active_set.tolist().index(s) for s in paI if s in msm.active_set]
    paII_prime = [msm.active_set.tolist().index(s) for s in paII if s in msm.active_set]

    tpt = pyemma.msm.tpt(msm, inactive_prime, paI_prime)
    MFPT_inactive_paI[j,0] = tpt.mfpt/10000
    tpt = pyemma.msm.tpt(msm, paI_prime, inactive_prime)
    MFPT_inactive_paI[j,1] = tpt.mfpt/10000  #inactive to partially active I(and the opposite)

    tpt = pyemma.msm.tpt(msm, paI_prime, paII_prime)
    MFPT_paI_paII[j,0] = tpt.mfpt/10000
    tpt = pyemma.msm.tpt(msm, paII_prime, paI_prime)
    MFPT_paI_paII[j,1] = tpt.mfpt/10000      #partially active state I to partially active state II (and the opposite)


print('transition  mean  error \n')

print("inactive -> partially active state I " + str(np.mean(MFPT_inactive_paI[:,0]))  + ' ' + str(np.std(MFPT_inactive_paI[:,0])))
print("partially active state I -> inactive " + str(np.mean(MFPT_inactive_paI[:,1]))  + ' ' + str(np.std(MFPT_inactive_paI[:,1])))

print("partially active state I -> partially active state II " + str(np.mean(MFPT_paI_paII[:,0]))  + ' ' + str(np.std(MFPT_paI_paII[:,0])))
print("partially active state II -> partially active state I " + str(np.mean(MFPT_paI_paII[:,1]))  + ' ' + str(np.std(MFPT_paI_paII[:,1])))
