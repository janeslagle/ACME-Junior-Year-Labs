# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
Jane Slagle
Math 321 Section 2
10/22/21
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree  #need for problem 5
from scipy import stats

# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    #use array broadcasting and the axis argument to avoid using a for loop
    #have array, vector: if one 1 dim and do matrix - vector: will subtract vector from each row in matrix which is what want
    #so do point - neighbor point and gives a matrix where each term shifted by the neighbor point
    
    distance = la.norm((X - z), axis = 1)  #will do each row and return each row where each row is the norm of that row in original matrix
                                         #don't want component wise norm: want rows instead
                                         
    minX = np.argmin(distance)           #equation 1.2
    minDistance = np.min(distance)          #equation 1.2
    
    return X[minX], minDistance          #want to return it as a list, so return as X[minX] and the minimum distance


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):    #the constructor of the class.
        """Constructor accepts a single parameter x."""
        
        if type(x) != (np.ndarray):
            raise TypeError("x parameter needs to be of type NumPy array")   #raise TypeError if x is not NumPy array]
            
        self.value = x      #save x as attribute called value
        self.left = None    #initialize attributes left, right and pivot as None
        self.right = None   #left and right are child noods
        self.pivot = None   #pivot assigned when node inserted into tree. the pivot gives you the index for where you're at
                            #pivot is the depth of the tree you are looking at and the index mod k (so whichever element in the array looking at
                            #on the specific level
           
           
# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        newNode = KDTNode(data)      #create new KDTNode containing x where x is the data
        newNode.pivot = 0            #set the new nodes pivot to be 0
        
        #number 1 in problem 3
        if self.root is None:        #check if tree is empty
            self.root = newNode      #assign root attribute to new node
            self.k = len(data)       #set k attribute as length of data
            return                   #need this so that will exit it since have an if statement here followed by another if statement
             
        current = self.root          #want to start at the root of the tree so make variable that is the root so that can start at it         
       
        if len(current.value) != len(data):      #check if the data to be inserted is in R^k or not. Will check if it is the correct dim or not
            raise ValueError("The data to be inserted is not in R^k")  #raise exception if length of data is not in tree 
            
        #numbers 2 and 3 in problem 3
        def find_and_link(current):
        
            if np.all(data) == current.value[current.pivot]:               #raise value error if the node containing data already in tree. these are arrays and want to check at those pivots
                raise ValueError("The node is already in the tree, no duplicates!")
            
            if data[current.pivot] < current.value[current.pivot]:  #check if the data adding at whatever pivot we are at is less than pivot value
                                                                    #of wherever we currently are in the tree. Then need to add the thing inserting
                                                                    #on the left of where we currently are
                if current.left == None:                            #if the spot is empty, add the new node there
                    current.left = newNode
                    current.left.pivot = (current.pivot + 1) % self.k    #update the index of the pivot
                else:                                               #if the spot is not empty, then recursively call the function on it
                    return find_and_link(current.left)              #put current.left because you want to start again where you just left off
                    
            else:   #if the data adding at whatever pivot we are at is greater than pivot value of whatever is currently there. Then add the new thing
                    #to the right of where we are in the tree
                if current.right == None:
                    current.right = newNode
                    current.right.pivot = (current.pivot + 1) % self.k
                else:
                    return find_and_link(current.right)
                       
        return find_and_link(self.root)    #start the recursive search at the root of the tree
                 

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        #follow the pseudo code given in Algorithm 1.1 in the lab manual
        
        def KDSearch(current, nearest, dist):
            if current == None:
                return nearest, dist
            
            else:
                x = current.value
                i = current.pivot
                d = np.linalg.norm(x - z)
                
                if d < dist:
                    nearest = current
                    dist = d
                    
                if z[i] < x[i]:
                    nearest, dist = KDSearch(current.left, nearest, dist)
                    
                    if z[i] + dist >= x[i]:
                        nearest, dist = KDSearch(current.right, nearest, dist)
                    
                else:
                    nearest, dist = KDSearch(current.right, nearest, dist)
                    
                    if z[i] - dist <= x[i]:
                        nearest, dist = KDSearch(current.left, nearest, dist)
                    
            return nearest, dist
            
        node, dist = KDSearch(self.root, self.root, np.linalg.norm(self.root.value - z))
            
        return node.value, dist
       
        
    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    
    def __init__(self, n_neighbors):
        """Constructor for the class that accpets an int n_neighbors, the number of neighbors
        to include in the vote.
        Attributes: n_neighbors"""
        
        self.n_neighbors = n_neighbors   #save n_neighbors as an attribute
        
    def fit(self, X_array, y_array):
        """Accept an m x k NumPy array X and a 1-D NumPy array y with m entries.
        Assigns the training set to its appropriate labels. Fitting the data to the labels.
        The X array is the training set and the y array are the training labels."""
        
        tree = KDTree(X_array)           #load SciPy KDTree with the data in X array
        
        self.tree = tree                 #save the tree and the labels as attributes
        self.label = y_array   #told in lab manual that array y are the labels for the tree
        
    def predict(self, z_array):
        """Accept a 1-D NumPy array z with k entries."""
        
        distances, indices = self.tree.query(z_array, k = self.n_neighbors)   #query the KDTree for the n_neighbors elements of X that are closest to z
        #want the target in the query function to be z so put array z into the query function
        #the k = neighbors: Scipy query returns distances, indices so indices they give you will be indices of labels 
        #so get labels at those indices (why have k)
       
        return stats.mode(self.label[indices])[0][0]
        #want to return most common label of those neighbors, so use scipy.stats function to do that
        #have an attribute for the labels, so use that and want the labels at those common indices and our y array (labels) is an array
        #so able to access it at indices index
        
  
# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load(filename)                    #first need to extract the data from the given file. Use the code given in lab manual to extract data
    X_train = data["X_train"].astype(np.float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float)
    y_test = data["y_test"]
     
    #load a classifier from problem 5 with data X_train, correspoding labels y_train:
    classifier = KNeighborsClassifier(n_neighbors)   #put n_neighbors in because that's what put in when you call problem 5
    
    classifier.fit(X_train, y_train)    #want classifier to have data X_train, y_train labels in it, so use fit to do that
    
    #use the classifier to predict labels of each image in X_test, use the predict function to do this:
    x = np.array([classifier.predict(k) for k in X_test]) 
    #predict returns a number, but want an array, so put it into an array
    #put it into an array (x) so that can compare it to all entries in y_test in the next part
    
    #find classification accuracy, the % of predictions that match y_test
    accuracy = 0
    j = 0
    for i in x:   #so loop through everything in x and compare every entry in x to every entry in y_test
        if i == y_test[j]:
            accuracy = accuracy + 1   #it's correct when they're equal, so add 1 to the accuracy
        j = j + 1
        
    accuracy = accuracy / len(y_test)   #to get the percentage
    
    return accuracy
        
    
