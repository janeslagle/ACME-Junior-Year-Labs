# qr_decomposition.py
"""Volume 1: The QR Decomposition.
Jane Slagle
Math 345 Section 2
10/21/21
"""

import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    #follow the psuedo code given in Algorithm 1.1 in the lab manual
    
    m, n = np.shape(A)       #store the dimensions of A
    Q = np.copy(A)           #make a copy of A with np.copy() function
    R = np.zeros((n, n))       #make an nxn array of all zeros
    
    for i in range(0, n):   #gets the range from i = 0 to n - 1
        R[i, i] = la.norm(Q[:, i])  
        Q[:, i] = Q[:, i] / R[i, i]         #normalize the ith column of Q
        
        for j in range(i + 1, n):  #gets range from i = i+1 to n-1
            R[i, j] = np.transpose(Q[:, j]) @ (Q[:, i])
            Q[:, j] = Q[:, j] - (R[i, j] * Q[:, i])     #orthognalize the jth column of Q
            
    return Q, R
    
    
# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    #use the QR decomposition of A and equation 1.1 in the lab manual to find |det(A)|
    #so use the QR decomposition function from problem 1
    #equation 1.1 says that |det(A)| = |det(Q)||product of R's diagonal entries|
    
    #np.prod finds product of array elements over a certain axis
    #np.diag gets the diagonals along certain axis in matrix
    
    Q, R = qr_gram_schmidt(A)
    
    return np.abs(np.prod(np.diag(R)))
    

# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    
    #need the next two lines for back subs part
    n = len(b)
    x = np.zeros(n)  #initialize x as an array of all zeros
    
    #use the QR decomposition to compute Q and R first:
    Q, R = qr_gram_schmidt(A)  #do this by using problem 1's function
    
    #now calculate y = tranpose(Q) * b:
    y = np.transpose(Q) @ b
    
    #use back subsitution to solve Rx = y:
    for k in range(n-1, -1, -1):
        x[k] = 1 / R[k,k] * (y[k] - sum(R[k, j] * x[j] for j in range(k + 1, n)))
    
    
    return x
    
   
#need this sign function for problems 4 and 5    
sign = lambda x: 1 if x>= 0 else -1
    
# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    #use psuedo code from Algorithm 1.2 in the lab manual
    
    m, n = np.shape(A)     
    R = np.copy(A)
    Q = np.identity(m)          #the mxm identity matrix
    
    for k in range(0, n):       #gets k in range from 0 to n-1
        u = R[k:,k].copy() 
        u[0] = u[0] + (sign(u[0]) * la.norm(u))    
        u = u / la.norm(u)    #normalize u
        R[k:, k:] = R[k:, k:] - 2 * np.outer(u, (np.transpose(u) @ R[k:, k:]))   
        Q[k:, :] = Q[k:, :] - 2 * np.outer(u, (np.transpose(u) @ Q[k:,:]))
        
    return np.transpose(Q), R
     
  
# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    #use pseudo code in Algorithm 1.3 given in lab manual
    
    m, n = np.shape(A)
    H = np.copy(A)
    Q = np.identity(m)
    
    for k in range(0, n - 2):  #gets k from 0 to n-3 (1 less than what the second value is in range)
        u = H[k + 1:, k].copy()
        u[0] = u[0] + (sign(u[0]) * la.norm(u))
        u = u / la.norm(u)
        
        H[k+1:, k:] = H[k+1:,k:] - 2 * np.outer(u, u.T @ H[k+1:,k:])
        H[:, k+1:] = H[:, k+1:] - 2 * np.outer(H[:, k+1:] @ u, u.T)
        Q[k+1:,:] = Q[k+1:,:] - 2 * np.outer(u, u.T @ Q[k+1:,:])
        
    return H, Q.T
    
   
