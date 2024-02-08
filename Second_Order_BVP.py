# 1D - CFD
# Julian Castrillon

'''Solver for a 1D Model Boundary Value Problem

The 1-D FEM functions are used to solve here a model BVP of the following form

d^2 u/dx^2 = f(x) \forall x \in [a, b]
u = u_a at x = a
u = u_b at x = b
'''
import numpy as np
from FEM_1D_Functions import GetGaussQuadraturePoints, GetGaussQuadratureWeights, LineElementShapeDerivatives,\
LineElementMappingGradient, LineElementIsoparametricMap, LineElementShapeFunction

# Bank of Functions ##############################################################################
##################################################################################################
def ComputeElementStiffness(ElementNodes, GaussOrder, PolDeg):
    """Compute element stiffness matrix for a simple second order BVP
    Inputs:
        ElementNodes(float-array): The nodal coordinates (global) of the element
        GaussOrder(int): The order of Gaussian quadrature needed
        PolDeg(int): The polynomial degree
    Outputs:
        k(float-array): The computed stiffness matrix for the element
    """
    EtaGauss = GetGaussQuadraturePoints(GaussOrder) # Etas
    wGauss   = GetGaussQuadratureWeights(GaussOrder) # Weights
    
    k = np.zeros((int(PolDeg+1),int(PolDeg+1))) # Initiate K matrix 
    
    for i in range(int(PolDeg)+1):
          
        for j in range(int(PolDeg)+1):
              
            k[i,j] = 0
                    
            for s in range(int(GaussOrder)):
                
                dphi   = LineElementShapeDerivatives(EtaGauss[s], PolDeg, int(i)) # Derivative of shape function
                dphj   = LineElementShapeDerivatives(EtaGauss[s], PolDeg, int(j)) # Derivative of shape function
                dxdEta = LineElementMappingGradient(EtaGauss[s], PolDeg, ElementNodes) # Jacobian
                   
                k[i,j] = k[i,j] + wGauss[s] * dphi * dphj * 1/(dxdEta)
                
    return k 
##################################################################################################    
def ComputeElementLoading(Function, ElementNodes, GaussOrder, PolDeg):
    """Compute element rhs loading vector for a simple second order BVP
    Inputs:
        Function(function): The handle to the function f(x) (implemented in the solver)
        ElementNodes(float-array): The nodal coordinates (global) of the element
        GaussOrder(int): The order of Gaussian quadrature needed
        PolDeg(int): The polynomial degree
    Outputs:
        f(float-array): The computed loading vector for the element
    """
    f = np.zeros((int(PolDeg+1),1)) # Initiate F matrix 
    EtaGauss = GetGaussQuadraturePoints(GaussOrder) # Etas
    wGauss   = GetGaussQuadratureWeights(GaussOrder) # Weights

    for i in range(int(PolDeg)+1):
       
        f[i,0] = 0
          
        for s in range(int(GaussOrder)):
           
            IsoMap = LineElementIsoparametricMap(EtaGauss[s], PolDeg, ElementNodes)          
            phi    = LineElementShapeFunction(EtaGauss[s], PolDeg, int(i)) # Derivative of shape function
            dxdEta = LineElementMappingGradient(s, PolDeg, ElementNodes)
            
            f[i,0] = f[i,0] + (wGauss[s] * Function(IsoMap) * phi * dxdEta)
            f
            
    return f
##################################################################################################
def AssembleGlobalStiffness(Nodes, Convty, GaussOrder, PolDeg):
    """Assemble the element stiffness matrices into a global stiffness matrix
    Inputs:
        Nodes(float-array): Numpy array with nodal coordinates
        Convty(int-array):  The connectivity matrix
        GaussOrder(int): The order of Gaussian quadrature needed
        PolDeg(int): The polynomial degree
    Outputs:
        K(float-array): The assembled global stiffness matrix
    """
    K = np.zeros((np.size(Nodes),np.size(Nodes)))
    s = np.size(Convty,1)

    for m in range(np.size(Convty,0)):
        
        ElementNodes = Nodes[int(Convty[m,0]):int(Convty[m,-1])+1]
        k = ComputeElementStiffness(ElementNodes, GaussOrder, PolDeg)
        
        R = np.zeros((np.size(Nodes),np.size(Nodes)))   
        R[m*(PolDeg):m*s+(s-m) , m*(PolDeg):m*s+(s-m)] = k[0:s,0:s] 
            
        K = K+R      
        
    return K
##################################################################################################
def AssembleGlobalLoading(Function, Nodes, Convty, GaussOrder, PolDeg):
    """Assemble the element rhs loading vectors into a global rhs vector
    Input:
        Function(function): The handle to the function f(x) (implemented in the solver)
        Nodes(float-array): Numpy array with nodal coordinates
        Convty(int-array):  The connectivity matrix
        GaussOrder(int): The order of Gaussian quadrature needed
        PolDeg(int): The polynomial degree
    Output:
        F(float-array): The assembled global rhs loading vector
    """
    F = np.zeros((np.size(Nodes),1))
    s = np.size(Convty,1)
    
    for m in range(np.size(Convty,0)):
    
        ElementNodes = Nodes[int(Convty[m,0]):int(Convty[m,-1])+1]
        f = ComputeElementLoading(Function, ElementNodes, GaussOrder, PolDeg)
        R = np.zeros((np.size(Nodes),1))  
        R[m*(PolDeg):m*s+(s-m)] = f[0:s] 
            
        F = F+R  
        
    return F
##################################################################################################
def ApplyDirichlet(K, F, DirNodes, DirVals):
    """Modify the matrix system by applying the Dirichlet boundary conditions
    Inputs:
        K(float-array): Global stiffness matrix
        F(float-array): Global rhs loading vector
        DirNodes(int):  List of node IDs where Dirichlet values are specified
        DirVals(float): List of Dirichlet values

    Outputs:
        K(float-array): The modified global stiffness matrix
        F(float-array): The modified global loading vector
    """
    for s in range(np.size(DirNodes)):
          
        i = DirNodes[s]
        F[:] = F[:] - np.vstack(K[:,i]) * DirVals[s]
        F[i] = DirVals[s]
        
        K[:,i] = 0
        K[i,:] = 0   
        K[DirNodes[s],DirNodes[s]] = 1
        
    return [K,F]
##################################################################################################
# END ############################################################################################