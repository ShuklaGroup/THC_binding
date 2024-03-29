# Why does delta-9-Tetrahydrocannbinoid act as a partial agonist for Cannabinoid Receptors?
This repository contains MSM feature file, final MSM object, codes and pdb structure files, and bootstrap samples used to genarate figures and calculations in the manuscript.

## features.pkl
This file can be downloaded from https://uofi.box.com/s/g1x5vfukl6492jtuzpx7u1rbyki2t85z. This file is a list of arrays where each array represents a MD trajectory. It contains 23 features for each MD trajectory used to build the final MSM. 

## final_MSM_obj.pkl
This file can be downloaded from https://uofi.box.com/s/hzzgzq0xw1tcgsjlnecklajpjdtmm4ma. It contains optimal MSM object with 200 clusters and 20 ns lagtime. 

## bootstraping
This folder can be downloaded from https://uofi.box.com/s/88nzwam5ygout7x3wvswxzlbcfkqxky3. It contains 200 bootstrap samples and corresponding MSMs with same state index as reference MSM.

## Figure_3
- Figure_3A.py: python code to generate two dimentional free energy landscape as shown in Figure 3A
- Figure_3B.py: python code to generate one dimentional free energy landscape as shown in Figure 3B
- Representative MD structures from different THC binding poses as shown in Figure 3C, 3D, and 3E.

## Figure_4
- Representative MD structures from different THC binding poses as shown in Figure 4.

## Figure_5
- Figure_5A.py: python code to generate two dimentional free energy landscape as shown in Figure 5A.
- Figure_5B.py: python code to generate TPT calculations as shown in Figure 5B.
- Representative MD structures from different macrostates as shown in Figure 5B.

## Figure_6
- Figure_6A.py: python code to generate two dimentional free energy landscape as shown in Figure 6A.
- Figure_6B.py: python code to generate TPT calculations as shown in Figure 6B.
- Representative MD structures from different TRP356 conformation as shown in Figure 6B.
- Figure_6C.py: python code to generate conditional probability calculations as shown in Figure 6C.

## Figure_7
- Figure_7A.py: python code to generate two dimentional free energy landscape as shown in Figure 7A.
- Figure_7B.py: python code to generate conditional probability calculations as shown in Figure 7B.
- Representative MD structures from partially active state.

## Figure_8
- Figure_8A.py: python code to generate 2-D scatter plot projecting the principal component 1 and 2 
- Figure_8B.py: python code for calculation of K-L divergence of partial agonist and agonist bound ensemble 
- Figure_8C.py: python code to generate 2-D scatter plot projecting toggle switch movement and TM6 movement 
- Figure_8D.py: python code to generate 2-D scatter plot projecting toggle switch movement and NPxxY movement
- agonist_heavy_atom_distances.npy: array containing closest heavy atom distances for every pair of residue in the agonist bound ensemble. This file can be downloaded from https://uofi.box.com/s/y92093ay6t8ma0p12jgyxpwxxjf2ibh8
- partial_agonist_heavy_atom_distances.npy: array containing closest heavy atom distances for every pair of residue in the partial agonist bound ensemble. This file can be downloaded from https://uofi.box.com/s/y92093ay6t8ma0p12jgyxpwxxjf2ibh8

## Mutual_information_calculation
- mdentropy_calculation.py : mutual information calculation for partial agonist and agonist bound ensemble 
- dijksta_graph.py : estimation of shorest allosteric path between toggle switch and NPxxY region for partial agonist and agonist bound ensemble 
- necessary files for the analysis can be downloaded from https://uofi.box.com/s/mm3yvufwfkkovja78ziv4jt1kzgw6fjn
