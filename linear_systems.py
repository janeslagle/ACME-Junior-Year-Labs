# linear_systems.py
"""Volume 1: Linear Systems.
Jane Slagle
Math 345 Section 2
10/7/21
"""

import numpy as np
import time
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla


# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    
    #rows are first, then columns when access matrix A, so j represents columns here and i represents rows here
    
    n = len(A)
    
    for j in range(n):   #want go through each column in matrix 
        for i in range(j + 1, n):   #within the column we're working in, go down through each row in that column so that can make all entries 0 in it
                                    #loop through each row, within the above column
            A[i,j:] = A[i, j:] - (A[i,j] / A[j,j]) * A[j,j:]    #reducing the column to zeros below the diagonal
                                                                #the colons : mean go through the whole row (which means do each column)
            
    return A
          
# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    
    #follow the psuedo code given in lab manual
    
    n = A.shape[0]  #number of columns
    U = A.copy()
    L = np.identity(n)
    
    for j in range(n):   #gets j = 0 to n - 1
        for i in range(j + 1, n):     #gets i = from j + 1 to m - 1 like want
            L[i,j] = U[i,j] / U[j,j]
            U[i,j:] = U[i,j:] - L[i,j] * U[j,j:]
            
    return L, U
    

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    
    n = len(b)
    L, U = lu(A)
    y = np.copy(b)
    x = np.zeros(n)
    
    #then just make the equations they tell you to in the lab manual to do this
    
    #equation 1.1
    for k in range(n):
        y[k] = b[k] - sum([L[k,j] * y[j] for j in range(k)])  #use list comprehension to loop through the summation in the equation
    
    #initialize x with thing given in lab manual
    x = [(1 / U[n,n]) * y[n] for n in range(n)]
    
    #equation 1.2
    for k in range(n-1, -1, -1):
        x[k] = 1/ U[k,k] * (y[k] - sum(U[k,j] * x[j] for j in range(k + 1, n)))
      
      
    """The next three lines are for testing problem 3
    print(b)   #use this to test it because solving Ax=b, so in theory here Ax should equal b (the 2 sides of the equation should be equal)
    print(np.matmul(A,x))"""
      
    return x
  


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    
    times_1 = []   #array with times for number 1 so that can plot it later
    times_2 = []
    times_3 = []
    times_4 = []
    
    
    #1 in the problem
    for n in range(1, 1000, 10):  #want to do for various values of n
        A = np.random.random((n,n))        #generate random nxn matrix A
        b = np.random.random((n, 1))       #generate random n-vector b
        start = time.time()                #start the time
        inverse_A = la.inv(A)              #invert A
        np.dot(inverse_A, b)               #left multiply inverse of A to be
        end = time.time() - start
        times_1.append(end)                 #add the times on to the array of times have
        
    #2 in the problem    
    for n in range(1, 1000, 10):
        A = np.random.random((n,n))
        b = np.random.random((n,1))
        start = time.time()
        la.solve(A, b)
        end = time.time() - start
        times_2.append(end)
    
    #3 in the problem    
    for n in range(1, 1000, 10):
        A = np.random.random((n,n))
        b = np.random.random((n,1))
        start = time.time()
        L, P = la.lu_factor(A)   #from lab manual
        la.lu_solve((L,P), b)  #from lab manual
        end = time.time() - start
        times_3.append(end)
    
    #4 in the problem
    for n in range(1, 1000, 10):
        A = np.random.random((n,n))
        b = np.random.random((n,1))
        L, P = la.lu_factor(A)   #dont want to time this part
        start = time.time()      
        la.lu_solve((L,P), b)    #want to time this part
        end = time.time() - start
        times_4.append(end)
        
    #now plot everything:
    plt.title("Timing different scipy.linalg functions")   #give the graph a title
    
    plt.plot(times_1, 'r-', label = "left multiply inverse")   #plot the times for each method
    plt.plot(times_2, 'b-', label = "solve Ax=b")
    plt.plot(times_3, 'g-', label = "LU decomposition")
    plt.plot(times_4, 'm-', label = "factorization")
    
    plt.xlabel("n")                 #label the axes
    plt.ylabel("execution times")
    
    plt.legend(loc = "upper left")   #create a legend (key) for the graph
   
    plt.show()
    

# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    
    I = np.identity(n)
    
    #for creating B:
    diagonals = [1,-4,1]   #want 1 and 4 on the diagonals 
    offset = [-1,0,1]     #want 1 at offset of -1, 4 at offset of 0 and 1 at offset of 1 when make matrix B
    B = sparse.diags(diagonals, offset, shape = (n,n)).toarray()   #create B now

    list_A = []  #initialize list for creating A. the list will have B or I in it. is collection of all the lists
    
    for i in range(n):   #first row of A
        lst = [None] * n    #first creating list that is full of nones originally, every time go through loop, filling it with nones
        if i == 0:    #for the first row only, so only go through it once. 1st row is B, I and a bunch of nones
            lst[i] = B
            lst[i + 1] = I   #put B, I in correct places and nones will already be there bc start off with filled with nones
        elif i != (n - 1):  #everywhere have I B I, except for last row (just I B). for everything except for last row
            lst[i - 1] = I  #entry right before diagonal
            lst[i] = B      #entry at diagonal
            lst[i + 1] = I   #entry after diagonal
        else:  #now on the last row
            lst[i - 1] = I   #second to last entry
            lst[i] = B       #last entry is B (B always on diagonal)
            
        list_A.append(lst)   #build list_A each time loop through the for loop
            
    A = sparse.bmat(list_A)   #now create A as a sparse matrix. put list_A into it because .bmat takes a list in it
    
    #print(A.toarray())  #use this line to test if A is correct or not
    
    return A
   

# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    
    times_1 = []  
    times_2 = []
    
    for n in range(10, 80):   #want to do for various values of n
        A = prob5(n)   #call problem 5 to make matrix A
        b = np.random.random(n ** 2)  
        
        #solve Ax=b with approach 1 
        Acsr = A.tocsr()   #convert A to CSR format, call the CSR formatted A Acsr
        start_1 = time.time()   #start the time after do conversion
        spla.spsolve(Acsr, b)    #solve the system
        end_1 = time.time() - start_1
        times_1.append(end_1)
        
        
        #solve Ax=b with approach 2
        A_array = A.toarray()   #convert A to numpy array
        start_2 = time.time()
        la.solve(A_array, b)
        end_2 = time.time() - start_2
        times_2.append(end_2)
        
        
    plt.title("Time regular, sparse linear system solvers")
    
    plt.plot(times_1, 'r-', label = "sparse system solver") 
    plt.plot(times_2, 'b-', label = "linear system solver")
    
    plt.legend(loc = "upper left")
  
    plt.show()
    
        
