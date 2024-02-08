# Project_part_2
# CFD
# Julian Castrillon

import numpy as np
import matplotlib.pyplot as plt
from FEM_1D_Functions import GenerateMeshNodes, GenerateMeshConnectivity, ReferenceElementNodes,\
LineElementShapeFunction, LineElementShapeDerivatives

a = 0 # Lower bound
b = 1 # Upper bound
Size = 0.1 # Size of mesh
PolDeg = 4 # Polynomial order
Eta = np.linspace(-1,1,100) # Eta interval for plotting
Domain = np.array([a,b]) # Domain interval

[NumElmt, NumNodes, Nodes] = GenerateMeshNodes(Domain, PolDeg, Size, ReturnNumElmt=True, ReturnNumNodes=False)
Convty    = GenerateMeshConnectivity(NumElmt, PolDeg) # Connectivity matrix
RefNodes  = ReferenceElementNodes(PolDeg)
ShapeVal  = np.zeros([np.size(Eta),PolDeg+1]) # Shape values
dShapeVal = np.zeros([np.size(Eta),PolDeg+1])# Shape values derivatives

for LocalNode in range(PolDeg+1):
    
    for i in range(np.size(Eta)):
          
        ShapeVal[i,LocalNode]  = LineElementShapeFunction(Eta[i], PolDeg, LocalNode)
        dShapeVal[i,LocalNode] = LineElementShapeDerivatives(Eta[i], PolDeg, LocalNode)

plt.figure(1)
plt.plot(Eta,ShapeVal,'-')
plt.title("Element shape values")
plt.xlabel("Reference")
plt.ylabel("Eta")
plt.grid()
plt.savefig('ShapVal.png')

plt.figure(2)
plt.plot(Eta,dShapeVal,'-')
plt.title("Element shape value derivatives")
plt.xlabel("Reference")
plt.ylabel("Eta")
plt.grid()
plt.savefig('dShapeVal.png')