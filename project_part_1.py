# Project_part_1
# CFD
# Julian Castrillon

from FEM_1D_Functions import GenerateMeshNodes, GenerateMeshConnectivity

Domain = [0,160]; # Domain Interval
PolDeg = 2; # Polinomial order
h = 1 # Mesh size

[NumElmt, NumNodes, Nodes] = GenerateMeshNodes(Domain, PolDeg, h, ReturnNumElmt=True, ReturnNumNodes=False)
Convty = GenerateMeshConnectivity(NumElmt, PolDeg)

print('\n')
print('Number of elements:\n', NumElmt, '\n\n')
print('Number of nodes:\n', NumNodes, '\n\n')
print('List of nodes:\n', Nodes, '\n\n')
print('Connectivity Matrix:\n', Convty, '\n\n')