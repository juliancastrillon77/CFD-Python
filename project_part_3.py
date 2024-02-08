# Project_part_3
# CFD
# Julian Castrillon

import numpy as np
from FEM_1D_Functions import GenerateMeshNodes, GenerateMeshConnectivity, GetGaussQuadraturePoints, GetGaussQuadratureWeights,\
LineElementMappingGradient, LineElementShapeDerivatives, LineElementIsoparametricMap, LineElementShapeFunction

a = 0 # Lower bound
b = 1 # Upper bound
Size = 0.1 # Size of elemn
PolDeg = 1 # Polynomial order
element = 1 # Element
Function = lambda x: x # Function

Domain     = np.array([a,b])
GaussOrder = 3#(PolDeg+1)/2 # Gauss order
EtaGauss   = GetGaussQuadraturePoints(GaussOrder) # Ettas
wGauss     = GetGaussQuadratureWeights(GaussOrder) # Weights
[NumElmt, NumNodes, Nodes] = GenerateMeshNodes(Domain, PolDeg, Size, ReturnNumElmt=True, ReturnNumNodes=False)
Convty = GenerateMeshConnectivity(NumElmt, PolDeg) # Connectivity Matrix
ElementNodes = Nodes[int(Convty[element,0]):int(Convty[element,-1])+1]

k = np.zeros((int(PolDeg+1),int(PolDeg+1))) # Initiate K matrix 
f = np.zeros((int(PolDeg+1),1)) # Initiate Loading vector 

for i in range(int(PolDeg)+1):
      
    for j in range(int(PolDeg)+1):
          
        k[i,j] = 0
        f[i,0] = 0
            
        for s in range(int(GaussOrder)):
            
            dxdEta = LineElementMappingGradient(EtaGauss[s], PolDeg, ElementNodes) # Jacobian
            
            dphi   = LineElementShapeDerivatives(EtaGauss[s], PolDeg, int(i)) # Derivative of shape function
            dphj   = LineElementShapeDerivatives(EtaGauss[s], PolDeg, int(j)) # Derivative of shape function
            k[i,j] = k[i,j] + wGauss[s] * dphi * dphj * 1/(dxdEta)
            
            IsoMap = LineElementIsoparametricMap(EtaGauss[s], PolDeg, ElementNodes)          
            phi    = LineElementShapeFunction(EtaGauss[s], PolDeg, int(i)) # Shape function
            f[i,0] = f[i,0] + (wGauss[s] * Function(IsoMap) * phi * dxdEta)

print('\n') 
np.set_printoptions(precision=5)
print('Local stiffnes matrix:\n\n', k, '\n\n')
np.set_printoptions(precision=5)
print('Local loading vector:\n\n', f, '\n\n') 