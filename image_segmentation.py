# image_segmentation.py
"""Volume 1: Image Segmentation.
Jane Slagle
Math 345 Section 2
11/2/21
"""

import numpy as np
import math
from scipy import linalg as la
from scipy import sparse as sp
from imageio import imread
import scipy.sparse.linalg as ssl
from matplotlib import pyplot as plt

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    #do equation 1.1 in the lab manual:
    #want diagonal entries of D, find by summing A over an axis (can either sum over axis = 0 or 1 since A is adjacency matrix for
    #undirected graph which means that A is symmetric matrix)
    
    D = np.sum(A, axis = 0)   #finds sums of all w entries in 1.1 formula
    D = np.diag(D)            #now make D a diagonal matrix
    
    #do equation 1.2 in the lab manual:
    L = D - A    
    
    return L
    

# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    #Firt find number of connected components in graph:
    #the number of zero eigenvalues of Laplacian matrix exactly equals number of connected components
    
    laplacian_A = laplacian(A)           #find laplacian matrix of A using function in problem 1
    eigvalues = la.eig(laplacian_A)      #use scipy.linalg.eig() function to find eigenvalues
    
    #count how many of eigenvalues of the laplacian matrix are equal to 0
    total = 0                            #counter variable that stores how many eigenvalues are 0
    
    #scipy eig function returns list of eigenvalues and matrix of eigenvectors. So to access eigenvalues, get 1st element 
    #of eigvalues by doing real_eigvalues[0] since list is returned before the matrix with the function
    
    #loop through all eigenvalues (so loop through eigvalues[0]) and increment total counter when have a 0 eigenvalue
    for i in eigvalues[0]:
        if i < tol:                      #all eigenvalues that are less than tol are 0
            total = total + 1
    
    #Find algebraic connectivity. This is second smallest eigenvalue of L
    sorted_eigs = np.sort(eigvalues[0])  #sort all eigenvalues. By default, they will be sorted in ascending order
                                         #do this at [0] because la.eig gives 2 things: eigenvalues 1st, eigenvectors 2nd.
                                         #only want sort eigenvalues
    alg_connect = sorted_eigs[1]         #algebraic connectivity will be 2nd smallest eigenvalue
    
    return total, alg_connect
    
    
# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)             #read the image
        scaled_image = image / 255           #scale image so that contains floats btw 0, 1 by dividing the image by 255
        self.image = scaled_image            #then store image as attribute
        
        self.m = scaled_image.shape[0]       #get the dimensions for matrix A in problem 4
        self.n = scaled_image.shape[1]       
        
        #if image in color part of problem:
        brightness = scaled_image            #initalize brightness as being scaled because will have if statement that checks if image colored
                                             #and if image is colored then will change the brightness. But if image grayscale, brightness will just
                                             #equal this 
        if len(brightness.shape) == 3:       #image is colored if its shape has 3 things in it (scaled represents scaled image here and
                                             #brightness equals scaled so this if statement is checking length of image shape
                                             
            brightness = scaled_image.mean(axis = 2)   #if image colored want brightness to be 2D array that averages RGB values at each pixel
            
        flattened = np.ravel(brightness)     #flatten brightness matrix into 1D array using np.ravel function
        self.flattened = flattened           #store flattened brightness as an attribute
        
       
    # Problem 3
    def show_original(self):
        """Display the original image."""
        
        #plot image with grayscale parameter added because if it is colored, it will override the grayscale stuff
        plt.imshow(self.image, cmap = "gray")
        plt.show()
        
        """Test this by running ImageSegmenter("dream.png").show_original() in terminal"""
        
    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):  
       
        #initialize A as a scipy.sparse.lil_matrix 
        A = sp.lil_matrix((self.m * self.n, self.m * self.n))   #A is mn x mn matrix
        
        #want to save D as vector with mn entries
        D = np.zeros(self.m * self.n)     #initialize D as array with mn entries, with all entries 0.
                                          #will have for loop later where change entries of D to be the weights 
                       
        #do step 2 in lab manual:
        for i in range(self.m * self.n):  #loop through each vertex from 0 to mn-1
            #use get_neighbors to complete step 2a in lab manual
            #get_neighbors returns neighbors indices of current vertex i and how far away they are from i
            vertices, distances = get_neighbors(i, r, self.m, self.n)   #i is index want to get neighbor stuff for
            
            #step 2b in lab manual:
            #find weights using equation 1.3 and store them in A
            #B is the brightness which have saved as an attribute, self.flattened
            
            B_i = self.flattened[i]   #brightness attribute is an array
            
            weights = []              #store all weights in list
            
            #vertices is a list of j values, so need to loop through all of the js to find weights
            #first find B_j. Everything else after B_j will be computed in same for loop for j because everything else 
            #depends on what that j value is
            
            for j, distance in zip(vertices, distances):    #doing zip loops through vertices, distances array at same time
                B_j = self.flattened[j]                     
                difference = np.abs(B_i - B_j)
                
                #get the two terms subtracting in equation 1.3
                first_term = (-1 * difference) / sigma_B2
                second_term = distance / sigma_X2 
            
                #now do the actual equation in 1.3:
                weight = np.exp(first_term - second_term)
                weights.append(weight)                     #add the weight to our list of weights
            
                #store weights in A. Use hint given in problem
                A[i, j] = weight
            
            #now do step 2c
            D[i] = np.sum(weights)
            
        A = sp.csc_matrix(A)
        
        return A, D
     
             
    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        
        #number 1 in the problem:
        L = sp.csgraph.laplacian(A)
        
        #number 2 in the problem:
        #construct D^(-1/2) as sparse diagonal matrix using D and scipy.sparse.diags()
        D_half = sp.diags(1/np.sqrt(D)).tocsc()   #says in lab manual that D^(-1/2) is 1/sqrt(dij)
        
        product = (D_half @ L) @ D_half
        
        #number 3 in the problem:
        product_eigenvector = ssl.eigsh(product, which = "SM", k = 2)[1][:,1] 
        
        #eigsh returns eigenvalues 1st, eigenvectors 2nd. [1] returns eigenvectors only
        #[:,1] #access the 1th column because doing k=2 and stuff computes the 2 smallest and we want the second-smallest only
        #so do[:,1] to make sure that only return second smallest one and not 2 things
        
        #number 4 in the problem:
        #reshape the eigenvector as m x n matrix
        product_eigenvector = product_eigenvector.reshape((self.m,self.n))
        
        #use matrix to construct desired boolean mask
        mask = product_eigenvector > 0      #desired mask here are the things that are positive
        
        return mask
       

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        
        #call methods from problems 4, 5 to get the segmentation mask
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A, D)
        
        #Need to apply masks to the images:
        #calculate pos, neg masks depending on if image is in color or not in color
        #if colored: shape is 3. if not colored: shape is 2
        #so shape will change depending on whether image is colored or not
        #if colored: can't just multiply by mask so you have change it depending on that
        
        if len(self.image.shape) == 2:    #if grayscale
            pos = self.image * mask       #apply the mask to image
            neg = self.image * ~mask      
        elif len(self.image.shape) == 3:  #if colored
            pos = np.multiply(self.image, np.dstack((mask, mask, mask)))      #given in the lab manual
            neg = np.multiply(self.image, np.dstack((~mask, ~mask, ~mask)))
            
        #want to plot them all as subplots:
        
        #plot the original:
        plt.subplot(1,3,1)
        plt.imshow(self.image, cmap = "gray")   #put cmap = gray because if its color, it will override it anyway
        
        #plot the positive:
        plt.subplot(1,3,2)
        plt.imshow(pos, cmap = "gray")
        
        #plot the negative:
        plt.subplot(1,3,3)
        plt.imshow(neg, cmap = "gray")
        
        plt.show()
      
    
 # if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
