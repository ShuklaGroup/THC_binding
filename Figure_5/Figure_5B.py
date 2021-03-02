import numpy as np
import glob
import pickle
import pyemma

#######################################Initialization##############################################
#Macrostates (consist of microstates) with positive THC sidechain dihedral
one = [29, 47, 25, 8, 176]
two = [161, 6, 148, 136, 80]
three = [67, 100, 153, 91, 134]
four = [107, 128, 115, 39, 135]

#Macrostates (consist of microstates) with negative THC sidechain diheral
one_dash = [188, 44, 12, 127, 185]
two_dash = [56, 76, 23, 16, 93]
three_dash = [46, 45, 174, 59, 27]
four_dash = [82, 2, 168, 159, 94]

#initialization of list of lists {first list:forward movement(e.g. one to one desh) and backward movement(e.g. one_desh to one)
MFPT_1_1d = [[],[]]  
MFPT_1_2 = [[],[]]
MFPT_1d_2d = [[],[]]
MFPT_2_2d = [[],[]]
MFPT_2_3 = [[],[]]
MFPT_2d_3d = [[],[]]
MFPT_3_3d = [[],[]]
MFPT_3_4 = [[],[]]
MFPT_3d_4d = [[],[]]
MFPT_4_4d = [[],[]]

#####################################TPT calculations for each bootstrap sample####################
for j in range(200):
    msm = pickle.load(open("./../bootstraping/bt_80_" + str(j) +"_msm.pkl",'rb'))
    
    #indexes of the macrostates in the transition probability matrix of bootstrap sample
    one_prime = [msm.active_set.tolist().index(s) for s in one if s in msm.active_set]
    two_prime = [msm.active_set.tolist().index(s) for s in two if s in msm.active_set]
    three_prime = [msm.active_set.tolist().index(s) for s in three if s in msm.active_set]
    four_prime = [msm.active_set.tolist().index(s) for s in four if s in msm.active_set]

    one_hat = [msm.active_set.tolist().index(s) for s in one_dash if s in msm.active_set]
    two_hat = [msm.active_set.tolist().index(s) for s in two_dash if s in msm.active_set]
    three_hat = [msm.active_set.tolist().index(s) for s in three_dash if s in msm.active_set]
    four_hat = [msm.active_set.tolist().index(s) for s in four_dash if s in msm.active_set]

    if len(one_prime) > 0:                      #In some bootstrap sample all of the five microstates of macrostate one doesn't exist
        #print(one_prime,two_prime,msm.active_set)
        tpt = pyemma.msm.tpt(msm, one_prime, one_hat)
        MFPT_1_1d[0].append(tpt.mfpt/10000)
        tpt = pyemma.msm.tpt(msm, one_hat, one_prime)
        MFPT_1_1d[1].append(tpt.mfpt/10000)    #one to one_dash (and the opposite)
        
        tpt = pyemma.msm.tpt(msm, one_prime,two_prime)
        MFPT_1_2[0].append(tpt.mfpt/10000)
        tpt = pyemma.msm.tpt(msm, two_prime, one_prime)
        MFPT_1_2[1].append(tpt.mfpt/10000)    #one to two (and the opposite)

        tpt = pyemma.msm.tpt(msm, one_hat, two_hat)
        MFPT_1d_2d[0].append(tpt.mfpt/10000)
        tpt = pyemma.msm.tpt(msm, two_hat, one_hat)
        MFPT_1d_2d[1].append(tpt.mfpt/10000)  #one_dash to two_dash (and the opposite)

    tpt = pyemma.msm.tpt(msm, two_prime, two_hat)
    MFPT_2_2d[0].append(tpt.mfpt/10000)
    tpt = pyemma.msm.tpt(msm, two_hat, two_prime)
    MFPT_2_2d[1].append(tpt.mfpt/10000)      #two to two_dash (and the opposite)

    tpt = pyemma.msm.tpt(msm, two_prime, three_prime)
    MFPT_2_3[0].append(tpt.mfpt/10000)
    tpt = pyemma.msm.tpt(msm, three_prime, two_prime)
    MFPT_2_3[1].append(tpt.mfpt/10000)      #two to three (and the opposite)

    tpt = pyemma.msm.tpt(msm, two_hat, three_hat)
    MFPT_2d_3d[0].append(tpt.mfpt/10000)
    tpt = pyemma.msm.tpt(msm, three_hat, two_hat)
    MFPT_2d_3d[1].append(tpt.mfpt/10000)    #two_dash to three_dash (and the opposite)

    tpt = pyemma.msm.tpt(msm, three_prime, three_hat)
    MFPT_3_3d[0].append(tpt.mfpt/10000)
    tpt = pyemma.msm.tpt(msm, three_hat, three_prime)
    MFPT_3_3d[1].append(tpt.mfpt/10000)     #three to three_dash (and the opposite)

    tpt = pyemma.msm.tpt(msm, three_prime, four_prime)
    MFPT_3_4[0].append(tpt.mfpt/10000)
    tpt = pyemma.msm.tpt(msm, four_prime, three_prime)
    MFPT_3_4[1].append(tpt.mfpt/10000)      #three to four (and the opposite)

    tpt = pyemma.msm.tpt(msm, three_hat, four_hat)
    MFPT_3d_4d[0].append(tpt.mfpt/10000)
    tpt = pyemma.msm.tpt(msm, four_hat, three_hat)
    MFPT_3d_4d[1].append(tpt.mfpt/10000)    #three_dash to four_dash (and the opposite)


    tpt = pyemma.msm.tpt(msm, four_prime, four_hat)
    MFPT_4_4d[0].append(tpt.mfpt/10000)
    tpt = pyemma.msm.tpt(msm, four_hat, four_prime)
    MFPT_4_4d[1].append(tpt.mfpt/10000)    #four to four_dash (and the opposite)



print('transition  mean  error \n')
print("1 -> 1' " + str(np.mean(MFPT_1_1d[0])) + ' ' + str(np.std(MFPT_1_1d[0])))
print("1' -> 1 " + str(np.mean(MFPT_1_1d[1])) + ' ' + str(np.std(MFPT_1_1d[1])))

print("1 -> 2 " + str(np.mean(MFPT_1_2[0])) + ' ' + str(np.std(MFPT_1_2[0])))
print("2 -> 1 " + str(np.mean(MFPT_1_2[1])) + ' ' + str(np.std(MFPT_1_2[1])))

print("1' -> 2' " + str(np.mean(MFPT_1d_2d[0])) + ' ' + str(np.std(MFPT_1d_2d[0])))
print("2' -> 1' " + str(np.mean(MFPT_1d_2d[1])) + ' ' + str(np.std(MFPT_1d_2d[1])))

print("2 -> 2' " + str(np.mean(MFPT_2_2d[0])) + ' ' + str(np.std(MFPT_2_2d[0])))
print("2' -> 2 " + str(np.mean(MFPT_2_2d[1])) + ' ' + str(np.std(MFPT_2_2d[1])))

print("2 -> 3 " + str(np.mean(MFPT_2_3[0])) + ' ' + str(np.std(MFPT_2_3[0])))
print("3 -> 2 " + str(np.mean(MFPT_2_3[1])) + ' ' + str(np.std(MFPT_2_3[1])))

print("2' -> 3' " + str(np.mean(MFPT_2d_3d[0])) + ' ' + str(np.std(MFPT_2d_3d[0])))
print("3' -> 2' " + str(np.mean(MFPT_2d_3d[1])) + ' ' + str(np.std(MFPT_2d_3d[1])))

print("3 -> 3' " + str(np.mean(MFPT_3_3d[0])) + ' ' + str(np.std(MFPT_3_3d[0])))
print("3' -> 3 " + str(np.mean(MFPT_3_3d[1])) + ' ' + str(np.std(MFPT_3_3d[1])))

print("3 -> 4 " + str(np.mean(MFPT_3_4[0])) + ' ' + str(np.std(MFPT_3_4[0])))
print("4 -> 3 " + str(np.mean(MFPT_3_4[1])) + ' ' + str(np.std(MFPT_3_4[1])))

print("3' -> 4' " + str(np.mean(MFPT_3d_4d[0])) + ' ' + str(np.std(MFPT_3d_4d[0])))
print("4' -> 3' " + str(np.mean(MFPT_3d_4d[1])) + ' ' + str(np.std(MFPT_3d_4d[1])))

print("4 -> 4' " + str(np.mean(MFPT_4_4d[0])) + ' ' + str(np.std(MFPT_4_4d[0])))
print("4' -> 4 " + str(np.mean(MFPT_4_4d[1])) + ' ' + str(np.std(MFPT_4_4d[1])))
