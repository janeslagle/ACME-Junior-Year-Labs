# condition_stability.py
"""Volume 1: Conditioning and Stability.
Jane Slagle
Math 347 Volume 1 Lab
1/31/22
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    #find condition number of matrix using equation 10.3 in lab manual:
    #so need largest and smallest singular values of matrix A given
    
    sing_vals = la.svdvals(A)   #scipy.linalg.svdvals() returns singular values in decreasing order in array, so 
                                 #max singular value is 1st entry, min singular value is last entry 
    
    max_sing = sing_vals[0]      #max singular value
    min_sing = sing_vals[-1]     #min singular value
  
    if min_sing == 0:            #want return infinity if smallest singular value is 0
        return np.inf
    else:
        return max_sing/min_sing       #do equation 10.3 in lab manual to get condition number of matrix A

# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    
    absol = 0                                            #need for step 3
    rel = 0
    
    for n in range(0, 100):                                  
        #step 1:
        r_i = np.random.normal(1, 1e-10, size = 21)  #there are 21 wilkinson coefficients, so want sample 21 times
        new_coeffs = w_coeffs * r_i                        #use hadamard product to replace each wilk coefficient
        
        #step 2:
        new_roots = np.roots(np.poly1d(new_coeffs))        #want plot the roots, so first need to get new roots
        w_roots = np.sort(w_roots)                       #sort them
        new_roots = np.sort(new_roots)
        
        real_coeffs = np.real(new_roots)                 #get real part of coefficients
        imag_coeffs = np.imag(new_roots)                 #get imaginary part of coefficients
        plt.scatter(real_coeffs, imag_coeffs, marker = ',', s = 3)  #plot real part on x axis, imag part on y axis
        
        #step 3:
        #use same method in example box in lab manual to get condition numbers:
        temp = la.norm(new_roots - w_roots, np.inf) / la.norm(r_i, np.inf)
        absol += temp  #absolute condition number
        rel += temp * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf)    #relative condition number
    
    plt.scatter(np.real(w_roots), np.imag(w_roots), s = 20, color = "red")
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.show()
        
    return absol/100, rel/100    #want return average of absolute, relative condition numbers

# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    #use equation 10.4 to estimate condition number:
    #make H matrix using code given in problem:
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags
    
    A_eigs = la.eigvals(A)      #get eigenvalues of A
    A_H_eigs = la.eigvals(A+H)  #get eigenvalues of A+H
    A_H_eigs = reorder_eigvals(A_eigs, A_H_eigs)    #need to reorder the perturbed eigenvalues
    
    #use equation 10.4 to get abs, relative condition numbers:
    K_hat = la.norm(A_eigs - A_H_eigs, ord = 2) / la.norm(H, ord = 2)
    K = (la.norm(A, ord = 2) / la.norm(A_eigs, ord = 2)) * K_hat
    
    return K_hat, K

# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    x_vals = np.linspace(domain[0], domain[1], res)
    y_vals = np.linspace(domain[2], domain[3], res)
    X, Y = np.meshgrid(x_vals, y_vals)
    empty_matrix = np.zeros_like(X)
    
    for i, x in enumerate(x_vals):                #loop over the grid and get M matrix each time for x, y values from the grid
        for j, y in enumerate(y_vals):
            M = np.array([[1, x], [y, 1]])
            empty_matrix[i, j] = eig_cond(M)[1]   #get the condition number for each x, y value
                 
    plt.pcolormesh(X, Y, empty_matrix, cmap = "gray_r")    #get condition number of M using problem 3 function
    plt.show()

# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    #use next 2 lines to set up system 10.5 in lab manual
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)
    
    #use approach 1 in lab manual to get the least squares solution (this is normal equations method):
    #use A matrix found above as A in the equation in approach 1, and see from equation 10.5 that b is the y's
    c_n1 = la.inv(A.T @ A) @ A.T @ yk        #see from equation 10.5 that the x values are the c values in formula
    
    #use approach 2 in lab manual: QR decomposition method
    Q, R = la.qr(A, mode = 'economic')[:2]    #only want the first two things returned from la.qr function
    c_n2 = la.solve_triangular(R, Q.T @ yk)
    
    #plot NOW:
    poly1 = np.polyval(c_n1, xk)              #want plot the polynomials, so need to create the polynomials
    poly2 = np.polyval(c_n2, xk)
    
    plt.plot(xk, poly1, color = "mediumvioletred", label = "Inverse Method")    #plot the polynomials
    plt.plot(xk, poly2, color = "royalblue", label = "QR Method")
    plt.scatter(xk, yk, color = "darkgreen", marker = '.', label = "data")      #plot the data points
    plt.title("Polynomial Approximations of data in stability_data.npy")
    plt.legend(loc = "best")
    plt.show()
    
    return la.norm((A @ c_n1) - yk, ord = 2), la.norm((A @ c_n2) - yk, ord = 2)   #return forward error of both approx
    
# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    n_vals = np.arange(5, 55, 5)  
    
    x, n = sy.symbols('x, n')
    integrate_me = (x**n)*(sy.exp(x-1))  
    I_n = ((-1)**n)*(sy.subfactorial(n) - (sy.factorial(n) / np.e))   #equation 10.6 in lab manual
    
    I1 = []      #list to store the sy.integrate() I(n) formulas
    I2 = []      #list to store equation 10.6 answers in
    error = []   #list to store the rel forward error in
    
    for N in n_vals:
        #step 1:
        sub_int = integrate_me.subs({n: int(N)})       #substitue the n value as an int into the integral formula   
        I = float(sy.integrate(sub_int, (x, 0, 1)))    #integrate w/ respect to x from 0 to 1
        I1.append(I)
        
        #step 2: 
        sub_In = float(I_n.subs({n: int(N)}))          #substitue the n value want in for n
        I2.append(sub_In)
        
    #step 3:
    #find relative forward error:
    for i in range(len(n_vals)):
        rel_fw_er = la.norm(I1[i] - I2[i]) / la.norm(I1[i])        #get rel forward error for each n value
        error.append(rel_fw_er)   
        
    #now plot rel forward error at each n value
    plt.yscale('log')
    plt.plot(n_vals, error, color = "deeppink")
    plt.xlabel("n")
    plt.ylabel("relative forward error")
    plt.title("Relative forward error at each n value")
    plt.show()
 
if __name__ == '__main__':
    """#test problem 1:
    #test with some random matrices to see if get same answer as np.linalg.cond():
    A = np.array([[1, 1], [0, 1], [-1, 1]])
    print("Prob 1 answer = " + str(matrix_cond(A)))
    print("np.linalg.cond answer = " + str(np.linalg.cond(A)))
    
    print("")
    B = np.array([[1000, 0], [55, 0.5]])
    print("Prob 1 answer = " + str(matrix_cond(B)))
    print("np.linalg.cond answer = " + str(np.linalg.cond(B)))
    
    print("")
    #check that orthonormal matrix gives 1
    J = np.identity(3)
    print("Prob 1 answer = " + str(matrix_cond(J)))
    print("np.linalg.cond answer = " + str(np.linalg.cond(J)))
    
    print("")
    #check that singular matrix gives infinity
    C = np.array([[2, 2, 2, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    print("Prob 1 answer = " + str(matrix_cond(C)))
    print("np.linalg.cond answer = " + str(np.linalg.cond(C)))"""
    
    pass
