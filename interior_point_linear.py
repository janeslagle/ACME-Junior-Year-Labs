# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
Jane Slagle
Volume 2 lab
3/29/22
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    #can do all problems in same function since they all need the same parameters
    def funky_func(x, lam, mu):
        #PROBLEM 1: function for vector-valued function F
        m, n = np.shape(A)
        M = np.diag(mu)      #make matrix M
        
        F = np.concatenate((A.T@lam + mu - c, A@x - b, M@x), axis = None)
        
        #PROB 2 stuff:
        DF = np.block([[np.zeros((n,n)), A.T, np.eye(n)], [A, np.zeros((m,m)), np.zeros((m,n))], [M, np.zeros((n,m)), np.diag(x)]])
        v = (x.T@mu)/n
        
        sigma = 0.1
        e = np.ones(n).T
        vec = np.concatenate([np.zeros(n), np.zeros(m), sigma*v*e])
        J = la.lu_solve(la.lu_factor(DF), (-1*F + vec))    #need to solve equation 9.2. gets us the search direction
               
        #PROB 3 stuff:
        #compute search direction: delta_x, delta_lam, delta_mu are what prob 2 solves for SO need to get results from prob 2
        del_x = J[:n]        #indexing comes from dimensions of block matrix for DF
        del_lam = J[n:m+n]
        del_mu = J[m+n:]
        
        #need mask for the conditions to get max alpha, delta
        mask_mu = del_mu < 0
        mask_x = del_x < 0 
        
        if np.all(del_mu):  
            alph_max = 1
        else:
            alph_max = min((-mu/del_mu)[mask_mu])   #apply mask
        
        if np.all(del_x):
            del_max = 1
        else:
            del_max = min((-x/del_x)[mask_x])
        
        alpha = min(1, 0.95*alph_max)
        delta = min(1, 0.95*del_max)
        
        return alpha*del_lam, alpha*del_mu, delta*del_x, v
        
    #PROB 4 stuff: want to use the subroutine stuff wrote for probs 1-3 to solve
    x, lam, mu = starting_point(A, b, c)
    for _ in range(niter):                #want run the iteration niter times
        _lamk, _muk, xk_, nu = funky_func(x, lam, mu)  #get the results from all the subroutine stuff
        x1 = x + xk_                      #follow the equations given
        lam1 = lam + _lamk
        mu1 = mu + _muk
        x = x1                            #update the x before iterate again
        (lam, mu) = (lam1, mu1)           #update before iterate again
      
        if nu < tol:
            break
        
    return x1, c.T*x1
 
#test probs 1-4:       
"""j, k = 7, 5
A, b, c, x = randomLP(j, k)
point, value = interiorPoint(A, b, c)
print(point[:k])
print(x)"""

def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    data = np.loadtxt(filename)   #first load the data from file
   
    #find the least absolute deviations line for data: use the boxes given in the examples
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]
    
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)
    
    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]          #beta is the slope
    b = sol[m+2*n] - sol[m+2*n+1]               #b is the y intercept
    
    #plot the LAD line with data set
    domain = np.linspace(0, 10, 200)
    plt.title("least absolute deviations line")
    plt.plot(domain, domain*beta + b, color = "mediumvioletred", label = "LAD line")
    plt.scatter(data.T[1], data.T[0], color = "darkslateblue", label = "data")   #plot the x values against the y values
    
    #plot the least squares solution
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    plt.plot(domain, domain*slope + intercept, color = "steelblue", label ="least squares solution")
    plt.legend(loc = "best")
   
    plt.show()
    
