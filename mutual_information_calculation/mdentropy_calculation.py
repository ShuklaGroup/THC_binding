import pickle 
import mdtraj as md 
import numpy as np 
import mdentropy
from mdentropy.metrics import DihedralMutualInformation
import glob
import sys



##############################################################################Transfer Entropy calculation for agonist bound#########################################
top = './traj/Agonist.prmtop'


for i,file in enumerate(glob.glob('./traj/agonist_bound_*.dcd')):
    t =  md.load(file,top=top)
    dihedral = DihedralMutualInformation(normed=True)
    transfer_en = dihedral.partial_transform(t)
    filen = file.split('./traj/')[1].split('.dcd')[0]
    pickle.dump(transfer_en,open('./pickle_files/' + filen + '_entropy.pkl','wb'))



##############################################################################Transfer Entropy calculation for partial agonist bound#########################################

topI = './traj/PA.prmtop'

for i,file in enumerate(glob.glob('./traj/PA_bound_*.dcd')):
    t =  md.load(file,top=topI)
    dihedral = DihedralMutualInformation(normed=True)
    transfer_en = dihedral.partial_transform(t)
    filen = file.split('./traj/')[1].split('.dcd')[0]
    pickle.dump(transfer_en,open('./pickle_files/' + filen + '_entropy.pkl','wb'))



