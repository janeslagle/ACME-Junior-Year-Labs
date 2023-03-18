# iterative_solvers.py
"""Volume 1: Iterative Solvers.
Jane Slagle
Volume 1 Lab
3/20/22
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy import sparse

# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot = False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    n = len(b) 
    x_k = np.zeros(n)             #need x^k to start off equation 9.2: x^k is vector of 0s initially
    abs_error = []                #store absolute error of approximation at each iteration
    iters = []                    #need keep track of num of iterations if need plot in prob 2
    
    #need make matrix D for equation 9.2:
    D_inv = np.zeros((n,n))       #initialize D^-1 as nxn matrix of all 0s (same size as A)
    A_diags = 1/A.diagonal()      #get 1 over all diagonal entries of A
    np.fill_diagonal(D_inv, A_diags)   #change diagonals of D^-1. This command will update D^-1 automatically
    
    for k in range(maxiter):      #only want iterate at most maxiter times
        x_k1 = x_k + D_inv@(b - A@x_k)
        abs_error.append(la.norm(A@x_k - b, ord = np.inf))  #add the abs error at each iteration to list
        iters.append(k+1)         #record the number of iteraions each time go through, add them all to list so that can plot over it
        if la.norm(x_k1 - x_k, ord = np.inf) < tol:
            break
        x_k = x_k1
        
    #create plot if plot variable is True
    if plot == True:
        plt.semilogy(iters, abs_error, color = "deeppink")
        plt.title("Convergence  of Jacobi Method plotted in deeppink")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
        
    return x_k1    #x_k1 is solution to system Ax=b via Jacobi method

# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    n = len(b)  
    x_k = np.zeros(n)       #need make x_k vector 
    abs_error = []          #list to store all of the errors in for when plot it
    iters = []              #list to store all of the iterations in for when plot it
     
    for k in range(maxiter):
        x_k1 = np.copy(x_k)  #need make a copy
        
        for i in range(n):   #i is ith row of A
            x_k1[i] = x_k[i] + (1/A[i, i])*(b[i] - A[i].T@x_k)   #equation 9.4 in lab manual
        
        abs_error.append(la.norm(A@x_k - b, ord = np.inf))  #add the abs error at each iteration to list
        iters.append(k+1)                                   #record the iterations each time loop through
        if la.norm(x_k1 - x_k, ord = np.inf) < tol:
            break
        x_k = x_k1
        
    if plot == True:         #create the plot if plot variable is true
        plt.semilogy(iters, abs_error, color = "deeppink")
        plt.title("Convergence  of Jacobi Method plotted in deeppink")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
        
    return x_k1
    
# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    n = len(b) 
    x_k = np.zeros(n)     
    
    A_diags = A.diagonal()   #get the diagonals from matrix A
    
    for k in range(maxiter):
        x_k1 = x_k.copy()    #need make a copy
        
        for i in range(n):   #i is the ith row of A
            # Get the indices of where the i-th row of A starts and ends if the
            # nonzero entries of A were flattened.
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
    
            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Aix = A.data[rowstart:rowend] @ x_k1[A.indices[rowstart:rowend]]
            
            x_k1[i] = x_k[i] + (1/A_diags[i])*(b[i] - Aix)   #equation 9.4 in lab manual
            
        if la.norm(x_k1 - x_k, ord = np.inf) < tol:
            break
        x_k = x_k1.copy()     #make the copy of it again
    
    return x_k1
    
# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    n = len(b)  
    x_k = np.zeros(n)       #need make x_k vector 
    converges = False       #boolean that says whether converges or not
    iterations = 0          #want keep track of number of iterations computed
    
    A_diags = sparse.csr_matrix.diagonal(A)   #get the diagonals from matrix A
     
    for k in range(maxiter):
        iterations += 1     #add to the iteraions each time loop through
        x_k1 = np.copy(x_k)  #need make a copy
        
        for i in range(n):   #i is ith row of A
            # Get the indices of where the i-th row of A starts and ends if the
            # nonzero entries of A were flattened.
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
    
            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Aix = A.data[rowstart:rowend] @ x_k1[A.indices[rowstart:rowend]]
            
            x_k1[i] = x_k[i] + (omega/A_diags[i])*(b[i] - Aix)   #equation 9.4 in lab manual
        
        if la.norm(x_k1 - x_k, ord = np.inf) < tol:
            converges = True    #if enter this for loop, means have converged
            break
        x_k = x_k1
        
    return x_k1, converges, iterations   

# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    #code from linear systems lab prob 5 to get matrix A:
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
    
    #make b vector:
    b_ = np.zeros(n)
    b_[0] = -100
    b_[-1] = -100
    
    #want to stack b on top of itself n times:
    b = np.tile(b_, n)
    
    #call problem 5 to solve Au=b: want return all 3 things from prob 5 function
    u = sor(A.tocsr(), b, omega, tol, maxiter)[0]
    converges = sor(A.tocsr(), b, omega, tol, maxiter)[1]
    iters = sor(A.tocsr(), b, omega, tol, maxiter)[2]
    
    #need make hotplate if plot variable is true
    if plot == True:
        u = np.reshape(u, (n,n)) #need reshape matrix u as nxn array before plot
        plt.pcolormesh(u, cmap = "coolwarm")
        plt.show()
        
    return u, converges, iters
   
# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    #run function from prob 6 with different omega values and n = 20:
    omegas = np.arange(1, 2, step = 0.05)
    computed_iters = []    #list to store all of the computed iteraions in
    
    for o in omegas:
        iters = hot_plate(n = 20, omega = o, tol=1e-2, maxiter=1000, plot=False)[2]
        computed_iters.append(iters)
    
    #plot the number of computed iterations as function of omega:
    plt.plot(omegas, computed_iters, color = "rebeccapurple")
    plt.show()
    
    return omegas[np.argmin(computed_iters)]   #return omega value that results in least number of iterations
 
def testing():
    """#test probs 1, 2:
    b = np.random.random(10)
    A = diag_dom(len(b))
    x_sol = jacobi(A, b, plot = True)
    print(np.allclose(A@x_sol, b))
    
    #test prob 3:
    x = gauss_seidel(A, b, plot = True)
    print(np.allclose(A@x, b))"""
    
    """#test prob 4:
    A = sparse.csr_matrix(diag_dom(10))   #need test it with a sparse matrix
    b = np.random.random(10)
    x = gauss_seidel_sparse(A, b)
    print(x)
    
    x_ = sor(A, b, 1.5)         #test prob 5 with same A, b as prob 4
    print(x_)
    print(np.allclose(A@x_[0], b))"""
    
    #test prob 6:
    hot_plate(16, 1, plot = True)
    
    
