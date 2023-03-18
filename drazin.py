# drazin.py
"""Volume 1: The Drazin Inverse.
Jane Slagle
Volume 1 Lab
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import csgraph as csg
import pandas as pd

# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    #check the 3 properties given to see if given Ad is indeed the Drazin inverse FOR REALS
    if np.allclose(A@Ad, Ad@A) and np.allclose(np.linalg.matrix_power(A, k+1) @ Ad, np.linalg.matrix_power(A, k)) and np.allclose(Ad@A@Ad, Ad):
        return True
    else:
        return False

# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    #use algorithm 10.1 to compute Drazin inverse A^D:
    (n,n) = np.shape(A)
    
    T1, Q1, k1 = la.schur(A, sort = lambda x: abs(x) > tol)  #need sorting function to sort Schur decomposition w/ 0 eigenvalues last
    T2, Q2, k2 = la.schur(A, sort = lambda x: abs(x) <= tol )  #need sorting function to sort Schur decomp w/ 0 eigenvalues 1st  
    
    U = np.hstack((Q1[:,:k1], Q2[:,:n-k1]))   #create change of basis matrix
    U_inv = np.linalg.inv(U)
    V = U_inv@A@U                #find block diagonal matrix in equation 10.1
    Z = np.zeros((n,n))
    
    if k1 != 0:
        M_inv = np.linalg.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_inv
    
    return U@Z@U_inv
   
# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    n = len(A)
    L = csg.laplacian(A)      #calculate the laplacian L of adjacency matrix A
    R = np.zeros((n,n))   #initialize R as all 0s. Will fill in R in for loop below
    
    for j in range(n):
        L_copy = L.copy()     #create copy of L so that we don't change original L matrix since use it every time loop through
        L_copy[j] = np.eye(n)[j]  #comes from equation 10.4: in Laplician, need jth row of Laplacian replaced by jth row of identity matrix
        LD = drazin_inverse(L_copy)  #need to get the drazin inverse from equation 10.4 using prob 2
        R[j] = np.diag(LD)    #fill R matrix
    R = R - np.eye(n)         #subtract identity from R matrix bc only want values on non-diags
    
    return R
   
# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """ 
        pairs = pd.read_csv(filename).to_numpy()  #this reads everything in and formats the data from file in their indiv pairs 
                                                  #(the pairs of the 2 names that are each connected) w/ each name separate
        connected_dict = dict()      #make a dict of everything that's connected
        unique_names = set()         #set that will store all of the indiv, unique names in
        
        #now actually create the dict and unique names set:
        for row in pairs:
            if row[0] not in connected_dict.keys():  #create the name in the dict if not already in it
                connected_dict[row[0]] = set()
            connected_dict[row[0]].add(row[1])       #add the connection to the dict btw the 2 names
            
            for name in row:
                unique_names.add(name)   #add all names to the set. since set, won't add duplicate names to it
        self.names = list(unique_names)  #convert the set of unique names to be list. this is getting all of the names of the nodes of graph as ordered list
        
        #get index of where each name is in the list of node names. need this bc the index of where name appears in list will be the 
        #same index of where it appears in the adjacency matrix
        n = len(unique_names)
        adj = np.zeros((n,n))        #initialize the adjacency matrix as all 0s
        for i, name in enumerate(self.names):  #use enumerate bc need index of each name. names corresp to row in adj matrix
            if name in connected_dict.keys():  #if name is in keys of dict, it means that it has connections
                for connected in connected_dict[name]:     #now loop through everything that the name is connected to
                    j = self.names.index(connected)
                    adj[i,j] = 1      #make entry in matrix be 1 if connection btw the 2 is there
                    adj[j, i] = 1
        
        eff_res = effective_resistance(adj)  #get the effective resistance matrix
        
        self.adj = adj
        self.eff_res = eff_res

    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        if node not in self.names and node != None:   #first raise ValueError
            raise ValueError("Node is not in the network or node does not equal None")
            
        #part a) of problem: zero out entries that already have connection:
        eff_res_copy = self.eff_res.copy()   #copy it so that able change it
        eff_res_copy = eff_res_copy * (np.ones(self.adj.shape) - self.adj)
        
        #case for when we aren't given a node
        #in this case: want to return tuple with names of nodes btw which next link should occur
        if node == None:
            #need to find next link by finding min value of array that is nonzero
            ordering = np.where(eff_res_copy == np.min(eff_res_copy[np.nonzero(eff_res_copy)]))  #use np.where to find min value in array
            smallest = ordering[0][0]
            second_smallest = ordering[1][0]
            return self.names[smallest], self.names[second_smallest]
        
        #case for when we are given a node
        #in this case: want return name of node which should be connected to node next out of all other nodes in network
        else:
            j =self.names.index(node)     #get index of the node
            column = eff_res_copy[:, j]   #get the column
            ordering = np.where(column == np.min(column[np.nonzero(column)]))[0][0]
            return self.names[ordering]

    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        #add a link btw the 2 nodes given as parameters:
        if node1 not in self.names and node2 not in self.names:  #raise ValueError if either of nodes not in network
            raise ValueError("Either node1 or node2 is not in the graph, oh no!")
        
        #add link by updating the adjacency matrix, effective resistance matrix
        i = self.names.index(node1)  #getting index of 1st node
        j = self.names.index(node2)  #getting nidex of 2nd node
        
        #add connection, update the adj matrix and eff resistance matrix now:
        self.adj[i, j] = 1
        self.adj[j, i] = 1
        self.eff_res = effective_resistance(self.adj)
        
def testing():
   """#test prob 1:
   A = np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
   Ad = np.array([[1,-3,9,81],[0,1,-3,-18],[0,0,1,3],[0,0,0,0]])
   print(is_drazin(A, Ad, 1))
   
   B = np.array([[1,1,3],[5,2,6],[-2,-1,-3]])
   Bd = np.zeros((3,3))
   print(is_drazin(B, Bd, 3))"""
   
   """#test prob 2:
   A = np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
   print(drazin_inverse(A, tol=1e-4))
   
   B = np.array([[1,1,3],[5,2,6],[-2,-1,-3]])
   print(drazin_inverse(B, tol=1e-4))"""
   
   """#test prob 3: 
   #test w/ the first test function given in fig 10.2, with the straight line
   A = np.array([[0, 1, 0, 0],
                 [1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0]])
   print(effective_resistance(A))"""
   
   #test probs 4, 5:
   links = LinkPredictor()
   print(links.predict_link())
   print(links.predict_link('Carol'))
   print(links.predict_link('Alan'))
   print(links.predict_link('Alan'))
   next = links.predict_link('Alan')
   links.add_link('Alan', next)
   print(links.predict_link('Alan'))
   next = links.predict_link('Alan')
   links.add_link('Alan', next)
   print(links.predict_link('Alan'))
   
   
