import networkx as nx
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

################################################################################loading of mutual information for partial agonist#############################################
S = np.zeros([285,285])
for i in range(20):
    M = pickle.load(open('./pickle_files/PA_bound_'+str(i)+'_entropy.pkl','rb'))
    S += M

M_avg = S/20

index = pickle.load(open('protein_protein_atom_ind.pkl','rb'))


###############################################################################graph definition###########################################################
G = nx.Graph()
arr = []

for i in range(285):
    for j in range(285):
        if M_avg[i,j] > 0 and [i,j] in index:
            G.add_node(i)
            arr.append((i,j,M_avg[i,j]))

G.add_weighted_edges_from(arr)

k =nx.algorithms.shortest_paths.weighted.dijkstra_path(G,229,270,weight='weight')

print(k)



################################################################################loading of mutual information for agonist#############################################
S = np.zeros([285,285])
for i in range(20):
    M = pickle.load(open('./pickle_files/agonist_bound_'+str(i)+'_entropy.pkl','rb'))
    S += M

M_avg = S/20

index = pickle.load(open('protein_protein_atom_ind.pkl','rb'))


###############################################################################graph definition###########################################################
G = nx.Graph()
arr = []

for i in range(285):
    for j in range(285):
        if M_avg[i,j] > 0 and [i,j] in index:
            G.add_node(i)
            arr.append((i,j,M_avg[i,j]))

G.add_weighted_edges_from(arr)

k =nx.algorithms.shortest_paths.weighted.dijkstra_path(G,229,270,weight='weight')

print(k)


