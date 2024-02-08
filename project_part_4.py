# Project_part_4
# CFD
# Julian Castrillon

import numpy as np
from FEM_1D_Functions import GenerateMeshNodes, GenerateMeshConnectivity
from Second_Order_BVP import AssembleGlobalStiffness, AssembleGlobalLoading

a = 0 # Lower bound
b = 1 # Upper bound
Size = 0.5 # Size of elemn
PolDeg = 1 # Polynomial order
Function = lambda x: x

Domain     = np.array([a,b])
GaussOrder = 3#(PolDeg+1)/2 # Gauss order
[NumElmt, NumNodes, Nodes] = GenerateMeshNodes(Domain, PolDeg, Size, ReturnNumElmt=True, ReturnNumNodes=False)
Convty = GenerateMeshConnectivity(NumElmt, PolDeg)

K = AssembleGlobalStiffness(Nodes, Convty, GaussOrder, PolDeg)
F = AssembleGlobalLoading(Function, Nodes, Convty, GaussOrder, PolDeg)

print('\n')
np.set_printoptions(precision=1) 
print('Global stiffnes matrix:\n\n', K, '\n\n')
np.set_printoptions(precision=5) 
print('Global loading vector:\n\n', F, '\n\n')