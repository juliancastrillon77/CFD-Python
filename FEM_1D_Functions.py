# 1D - CFD
# Julian Castrillon

import sys
import numpy as np
import matplotlib.pyplot as plt

# Bank of Functions ##############################################################################
##################################################################################################
def GenerateMeshNodes(Domain, PolDeg, Size, ReturnNumElmt=True, ReturnNumNodes=False):
    """Generate the node coordinates array for the 1-dimensional mesh
    Inputs:
        Domain[a,b]: The lower bound and upper bound of the domain
        PolDeg(int): The polynomial degree of the finite element Used for interpolation
        Size(float): The size of each element in the mesh
        ReturnNumElmt(bool):  Set to True if number of elements is to be returned
        ReturnNumNodes(bool): Set to True if number of Nodes is to be returned
    Outputs:
        Nodes(float-array): Numpy array with nodal coordinates
        NumNodes(int): Number of Nodes (if ReturnNumNodes = True)
        NumElmt(int):  Number of elements (if ReturnNumElmt = True)
    """
    NumElmt  = int((Domain[1]-Domain[0])/Size)
    NumNodes = int((NumElmt*PolDeg)+1)        
    Nodes    = np.zeros(NumNodes)
            
    for i in range(NumNodes):
            
        Nodes[i] = Domain[0] + i*(Size/PolDeg)
    
    return [NumElmt, NumNodes, Nodes]
################################################################################################## 
def GenerateMeshConnectivity(NumElmt, PolDeg):
    """Generate the mesh element Convty matrix for the 1-dimensional mesh
    Inputs:
        NumElmt(int): The number of elements in the mesh
        PolDeg(int):  The polynomial degree
    Outputs:
        Convty(int-array): The conectivity matrix
    """
    Convty = np.zeros((NumElmt,PolDeg+1))
    s = 0

    for i in range(NumElmt):
            
        Convty[i,:] = np.arange(s,s+PolDeg+1)
        s = s+PolDeg
        
    return Convty
##################################################################################################
def ReferenceElementNodes(PolDeg):
    """Generate nodal coordinates for the 1-dimensional reference element in [-1,1]
    Inputs:
        PolDeg(int): The polynomial degree
    Outputs:
        refNodes(float-array): The reference element node locations (in [-1,1])
    """   
    RefNodes = np.linspace(-1, 1, PolDeg+1)  
    
    return RefNodes
##################################################################################################
def LineElementShapeFunction(Eta, PolDeg, LocalNode):
    """Compute the shape function value for a line element. Phi(Eta). Based on the Lagrange Polynomial
    Inputs:
        Eta(float): Coordinate in the local coordinate system where the shape function is evaluated
        PolDeg(int): The polynomial degree
        LocalNode(int): ID of the local node for which the shape function is evaluated
    Outputs:
        ShapeVal(float): Value of the shape function evaluated in local system
    """
    RefNodes = ReferenceElementNodes(PolDeg)
    ShapeVal = 1
    
    for k in range(PolDeg+1):

        if k != LocalNode: # Excluedes the local node
        
            Q = (Eta - RefNodes[k]) / (RefNodes[LocalNode] - RefNodes[k]) # Component with values, this is just one term of the Lagrange polynomial
            ShapeVal = ShapeVal * Q
            
    return ShapeVal
##################################################################################################
def LineElementShapeDerivatives(Eta, PolDeg, LocalNode):
    """Compute the shape function derivatives for a line element.
    Inputs:
        Eta(float):  Coordinate in the local coordinate system where shape function derivative is evaluated
        PolDeg(int): The polynomial degree
        LocalNode(int): ID of the local node for which the shape function is evaluated
    Outputs:
        dShapeVal(float): Value of the shape function derivative evaluated in local system
    """
    RefNodes = ReferenceElementNodes(PolDeg)
    
    Qt = 0 # This is the 'sum' part from the derivative of the Lagrange polynomial
    
    for k in range(PolDeg+1): 
    
        if LocalNode != k: # Excluedes the local node
        
                O = 1/(RefNodes[LocalNode] - RefNodes[k])
                 
                for s in range(PolDeg+1):
                        
                    if ((s != LocalNode) and (s != k)): # Excluedes the local node

                        O = O*((Eta - RefNodes[s]) / (RefNodes[LocalNode] - RefNodes[s])) # Component with values, this is just one term of the Lagrange polynomial
                        np.set_printoptions(precision=3)
                                           
                Qt = Qt + O
        
    dShapeVal = Qt # Full Lagrange polynomial derivative

    return dShapeVal
##################################################################################################    
def LineElementIsoparametricMap(Eta, PolDeg, ElementNodes):
    """Evaluate the isoparametric mapping of global-local coordinates
    Inputs:
        ElementNodes(float-array): Element nodal coordinates (in global system)
        PolDeg(int): The polynomial degree
        Eta(float):  Coordinate in the local system where mapping is evaluated
    Outputs:
        IsoMap(float): Mapped coodinate value using shape function interpolation
    """
    IsoMap = 0
    for i in range(PolDeg+1):
        
        LocalNode = int(i)
        ShapeVal  = LineElementShapeFunction(Eta, PolDeg, LocalNode) 
        IsoMap    = IsoMap + ShapeVal * ElementNodes[i]
    
    return IsoMap
##################################################################################################
def LineElementMappingGradient(Eta, PolDeg, ElementNodes):
    """Evaluate the Jacobian of the isoparametric mapping of global-local coordinates
    Inputs:
        ElementNodes(float-array): The nodal coordinates (global) of the element
        PolDeg(int): The polynomial degree
        Eta(float):  Coordinate in the local system where mapping is evaluated
    Outputs:
        dxdEta(float): Mapped coodinate value using shape function interpolation
    """
    dxdEta = 0
    for i in range(PolDeg+1):
        
        LocalNode = int(i)
        dShapeVal = LineElementShapeDerivatives(Eta, PolDeg, LocalNode) 
        dxdEta    = dxdEta + dShapeVal * ElementNodes[i]
    
    return dxdEta
##################################################################################################
def PlotSolutions(Nodes, U, L, k, alpha, Flag):
    """Create a plot of the solution over the linear mesh
    Inputs:
        Nodes(float-array): Element nodal coordinates (in global system)
        U(float-array): Element nodal solutions
    Outputs:
        A beautiful plot
    """        
    s = np.linspace(0,1,np.size(Nodes)) # Resolution
    AnalyticalSolution = np.zeros(np.size(s))

    for i in range(np.size(s)):
       
        x = s[i]    
        T1 = - (L/np.pi)**2 * np.cos(np.pi*k*x/L) - alpha*(1-k**2) / ((2*np.pi*k)**2) * np.sin(2*np.pi*k*x/L)
        T2 =  ((L/np.pi)**2 * np.cos(np.pi*k/L)   - alpha*(1-k**2) / ((2*np.pi*k)**2) * np.sin(2*np.pi*k/L)   - (L/np.pi)**2 + 1) * x
        T3 =   (L/np.pi)**2
         
        AnalyticalSolution[i] = + T1 + T2 + T3
    
    if Flag == 1:
        plt.figure(1)
        plt.plot(Nodes,U,'-h', color = [0.4940, 0.1840, 0.5560], label = 'CFD')
        plt.plot(s,AnalyticalSolution, 'k--', label = 'Analytical solution') 
        plt.title("Solution")
        plt.xlabel("Domain")
        plt.ylabel("U")
        plt.legend(loc="upper left")
        plt.grid()  
        plt.savefig('Solution.png')
    
    return AnalyticalSolution           
##################################################################################################
def GetGaussQuadratureWeights(GaussOrder):
    """Tabulation of weights for Gauss quadrature
    Inputs:
        GaussOrder(int): The polynomial degree
    Outputs:
        float-array: the scalar weights for quadrature evaluation
    """
    if GaussOrder == 1:
        return np.array([2.0])
    elif GaussOrder == 2:
        return np.array([1.0, 1.0])
    elif GaussOrder == 3:
        return np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    elif GaussOrder == 4:
        return np.array([0.652145, 0.347855, 0.652145, 0.347855])
    elif GaussOrder == 5:
        return np.array([0.236987, 0.478629, 0.568889, 0.478629, 0.236927])
    else:
        sys.exit("Only upto 5th PolDeg Gauss quadrature is implemented")
##################################################################################################
def GetGaussQuadraturePoints(GaussOrder):
    """Tabulation of integration points for Gauss quadrature
    Points returned on domain [-1.0,1.0]
    Inputs:
        GaussOrder(int): the order of Gaussian quadrature needed
    Outputs:
        float-array: the scalar 1-dimensonal coordinates for quadrature evaluation
    """
    if GaussOrder == 1:
        return np.array([0.0])
    elif GaussOrder == 2:
        return np.array([-0.57735, 0.57735])
    elif GaussOrder == 3:
        return np.array([-0.774597, 0.0, 0.774597])
    elif GaussOrder == 4:
        return np.array([-0.339981, -0.861136, 0.339981, 0.861136])
    elif GaussOrder == 5:
        return np.array([-0.90618, -0.538469, 0.0, 0.538469, 0.90618])
    else:
        sys.exit("Only upto 5th PolDeg Gauss quadrature is implemented")
##################################################################################################
# END ############################################################################################