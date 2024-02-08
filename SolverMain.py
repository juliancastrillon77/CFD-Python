# 1D - CFD
# Julian Castrillon

import numpy as np
from scipy.sparse.linalg import cg
from FEM_1D_Functions import GenerateMeshNodes, GenerateMeshConnectivity, PlotSolutions
from Second_Order_BVP import AssembleGlobalStiffness, AssembleGlobalLoading, ApplyDirichlet

##################################################################################################
# FEM inputs
Flag = 1
a = 0 # Lower bound of domain
b = 1 # Upper bound of domain
PolDeg = 1 # Polynomial order for Lagrange interpolation
h = 0.05 # Size of mesh element
GaussOrder = 4 # Order of Gaussian Quadrature for numerical integration

# Function input
L = 1 # Constant
k = 2 # Constant
alpha = 5 # Constant
Function = lambda x: -(k**2 * np.cos((np.pi*k*x)/L)) - (alpha*(1-k**2)*np.sin((2*np.pi*k*x)/L))
#Function = lambda x: -x
##################################################################################################
Domain = np.array([a,b]) # Full domain
[NumElmt, NumNodes, Nodes] = GenerateMeshNodes(Domain, PolDeg, h, ReturnNumElmt=True, ReturnNumNodes=False) # Domain properties
Convty = GenerateMeshConnectivity(NumElmt, PolDeg) # Connectivy matrix

DirNodes = np.array([0,NumNodes-1]) # Boundary condition locations *--INPUT--*
DirVals = np.array([0,1]) # Boundary condition values              *--INPUT--*

K = AssembleGlobalStiffness(Nodes, Convty, GaussOrder, PolDeg) # Global assembly matrix
F = AssembleGlobalLoading(Function, Nodes, Convty, GaussOrder, PolDeg) # Global loading vector
[K,F] = ApplyDirichlet(K, F, DirNodes, DirVals) # Modify K and F to account for boundary conditions
[U, status] = cg(K, F) # Solver, U -> Solution
U = np.vstack(U)

print('\n')
np.set_printoptions(precision=1) 
print('Global stiffnes matrix:\n\n', K, '\n\n')
print('Global loading vector:\n\n', np.around(F,3), '\n\n') 
np.set_printoptions(precision=3)
print('Solution: \n\n', U, '\n\n')    
An = PlotSolutions(Nodes, U, L, k, alpha, Flag)
An = np.vstack(An)
Error = np.sum(np.abs(U-An))
np.set_printoptions(precision=3)
print('Error: \n\n', Error, '\n\n')  