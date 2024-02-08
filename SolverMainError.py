# 1D - CFD
# Julian Castrillon

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from FEM_1D_Functions import GenerateMeshNodes, GenerateMeshConnectivity, PlotSolutions
from Second_Order_BVP import AssembleGlobalStiffness, AssembleGlobalLoading, ApplyDirichlet

##################################################################################################
# FEM inputs
Flag = 0
Error = np.zeros([7, 5])
w = np.array([2, 4, 8, 16, 32]) # Array for k values
q = np.array([0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005]) # Array for h values

for s in range(np.size(w)):
    
    k = w[s]
    
    for i in range(np.size(q)):
        
        h = q[i] # Mesh size
        a = 0 # Lower bound of domain
        b = 1 # Upper bound of domain
        PolDeg = 1 # Polynomial order for Lagrange interpolation
        GaussOrder = 4 # Order of Gaussian Quadrature for numerical integration
        
        # Function input
        L = 1 # Constant
        alpha = 5 # Constant
        Function = lambda x: -(k**2 * np.cos((np.pi*k*x)/L)) - (alpha*(1-k**2)*np.sin((2*np.pi*k*x)/L))
        
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
              
        An = PlotSolutions(Nodes, U, L, k, alpha, Flag)
        An = np.vstack(An)
        Error[i, s] = np.sum(np.abs(U-An))
print('\n Error Matrix [h x k]\n\n', Error, '\n\n')

plt.figure(1)
plt.grid()
plt.yscale('log')
plt.plot(q,Error[:,0],'-h', color = [0.4940, 0.1840, 0.5560])
plt.title("Mesh Size Error k = 2")
plt.xlabel("Mesh Size")
plt.ylabel("Error")
plt.savefig('05.png')

plt.figure(2)
plt.grid()
plt.yscale('log')
plt.plot(q,Error[:,1],'-h', color = [0.4940, 0.1840, 0.5560])
plt.title("Mesh Size Error k = 4")
plt.xlabel("Mesh Size")
plt.ylabel("Error")
plt.savefig('01.png')

plt.figure(3)
plt.grid()
plt.yscale('log')
plt.plot(q,Error[:,2],'-h', color = [0.4940, 0.1840, 0.5560])
plt.title("Mesh Size Error k = 8")
plt.xlabel("Mesh Size")
plt.ylabel("Error")
plt.savefig('005.png')

plt.figure(4)
plt.grid()
plt.yscale('log')
plt.plot(q,Error[:,3],'-h', color = [0.4940, 0.1840, 0.5560])
plt.title("Mesh Size Error k = 16")
plt.xlabel("Mesh Size")
plt.ylabel("Error")
plt.savefig('001.png')

plt.figure(5)
plt.grid()
plt.yscale('log')
plt.plot(q,Error[:,4],'-h', color = [0.4940, 0.1840, 0.5560])
plt.title("Mesh Size Error k = 32")
plt.xlabel("Mesh Size")
plt.ylabel("Error")
plt.savefig('0005.png')