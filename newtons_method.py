# newtons_method.py
"""Volume 1: Newton's Method.
Jane Slagle
Math 346
1/24/22
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    converges = False          #bool to check if converges or not (set to false bc only converges when if statment condition
                               #is satisified
    counter = 0                #count number of iterations do
    
    n = 0
    if np.isscalar(x0):        #check if doing scalar newtons method (which is what want for prob 1, 3) because if x is 
        n = 1                  #scalar then have scalar version, but if x is not scalar, then have vector version
    
    #everything for prob 1, 3:
    if n == 1:                 
        for k in range(maxiter):   #want iterate until reach max # iteration times
            counter += 1 
            x1 = x0 - (alpha*(f(x0)/ Df(x0)))   
            if np.linalg.norm(x1 - x0) < tol:   #check if converges, want stop iterating if does
                converges = True
                break 
            x0 = x1                #update xk to be x_k+1 for the next iteration
    
    #problem 5:
    elif n == 0:                   #n=0 when have vector version of newtons method
        for k in range(maxiter):   #want iterate until reach max # iteration times
            counter += 1 
            #do the third bullet point in lab manual to get x_k+1:
            yk = la.solve(Df(x0), f(x0))
            x1 = x0 - (alpha*yk)
               
            if np.linalg.norm(x1 - x0) < tol:   #check if converges, want stop iterating if does
                converges = True
                break 
            x0 = x1                       #update x_k to be x_k+1 for next iteration

    #x1 will be approximation of the zero (what get out of equation 9.3 after stop iterating) 
    return x1, converges, counter          

# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    #want use newtons method to find r, so once find, f, f', plug them into prob 1 function
    #find f, f is the equation we're given in lab manual set equal to 0
    f = lambda r: P2*(1 - (1+r)**(-N2)) - P1*((1+r)**N1 - 1)
    
    #find derivative of f:
    Df = lambda r: (P2*N2)*((1+r)**((-1*N2)-1)) - (N1*P1)*((1+r)**(N1-1))
    
    r = newton(f, 0.1, Df)   #told want use initial guess 0.1
    
    return r[0]              #only want to return the r that makes it equal 0
   
# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    #make alpha as np.linspace and then loop through all alpha values and call problem 1
    alpha_vals = np.linspace(0, 1, 1001)[1:]   #want alpha is (0, 1] so don't want to include 0, so [1:] says to start at 1st
                                               #element after 0 like want
    best_iters = 15                            #this is max number of iterations can have       
    
    iterations = []                              #list to store the iterations in
    for alpha in alpha_vals:
        zeros, converges, iters = newton(f, x0, Df, tol, maxiter, alpha) 
        if iters < best_iters:                 #check for alpha value that gives you lowest # of iterations
            best_iters = iters                 #update the best_iters for the next time go through for loop
            best_alph = alpha                     
        iterations.append(iters) 
    
    #plot alpha against number of iterations:
    plt.plot(alpha_vals, iterations, color = "darkmagenta")
    plt.xlabel("Alpha values")
    plt.ylabel("Iterations")
    plt.show()
    
    return best_alph

# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    gamma = 5
    beta = 1
    
    xlin = np.linspace(-1/4, 0, 50)    #rectangle to search over, for x values
    ylin = np.linspace(0, 1/4, 50)     #for y values
    
    #define the 2 equations need:
    f = lambda x: (gamma*x[0]*x[1] - x[0]*(1+x[1]), (-x[0]*x[1] + (beta-x[1])*(1+x[1])))   #make the 2 equations given into 1 function
    #Df is the Jacobian here
    Df = lambda x: np.array([[(4*x[1]) - 1, 4*x[0]], [-x[1], -x[0]-2*x[1]]])
    
    for x in xlin:                    #loop through the rectangle to search for the valid x0
        for y in ylin:
            x0 = np.array([x,y])      
            new_pt = newton(f, x0, Df, alpha = 1)[0]   #find newton when alpha = 1
            if np.allclose(new_pt, np.array([0,1])) or np.allclose(new_pt, np.array([0, -1])):  #check if the point from newton is (0,1) or (0,-1)
                newer_pt = newton(f, x0, Df, alpha = 0.55)[0]                                   #now get newton when alpha = 0.55
                if np.allclose(newer_pt, np.array([3.75, 0.25])):                               #check if the point is (3.75, .25)
                    return x0                                                                   #if both of the if statements were true, return x0
    
# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    #do step 1 in lab manual:
    x_real = np.linspace(domain[0], domain[1], res)   #get the real parts: make it have length res because want a res x res grid
    x_imag = np.linspace(domain[2], domain[3], res)   #told that the domain is [rmin, rmax, imin, imax] so can get imaginary part like this
    X_real, X_imag = np.meshgrid(x_real, x_imag)     
    X_0 = X_real + 1j*X_imag                          #get the actual grid now
    
    #do step 2, do NOT call prob 5 to do it:
    for k in range(iters):
        X_k = X_0 - f(X_0)/ Df(X_0)                #plug X_0 in from above
        X_0 = X_k
        
    #do step 3:
    Y = np.empty((res, res))     #initialize Y as empty matrix
    for i in range(res):
        for j in range(res):     #want to fill each spot in Y so need to loop over res
            Y[i, j] = np.argmin(abs(X_k[i,j] - zeros))   #want Y[i,j] to be index of zero closet to (ith, jth) entry of X_k
    
    #do step 4:
    plt.pcolormesh(x_real, x_imag, Y, cmap = "brg")
    plt.show()

if __name__ == '__main__':
    """#test problem 1:
    print(newton(lambda x: np.exp(x) - 2, 2, lambda x: np.exp(x)))"""
    
    """#test problem 2:
    N1 = 30
    N2 = 20
    P1 = 2000
    P2 = 8000
    print(prob2(N1, N2, P1, P2))"""
    
    """#test problem 3:
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    Df = lambda x: np.power(np.abs(x), -2./3) / 3
    print(newton(f, 0.01, Df))
    print(newton(f, 0.01, Df, alpha=.4))"""
    
    #test problem 4:
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    Df = lambda x: np.power(np.abs(x), -2./3) / 3
    print(optimal_alpha(f, .01, Df))
    
    """#test problem 5:
    f = lambda x: np.array([x[0]*x[1], x[1]*x[2], x[0]-x[1]])
    Df = lambda x: np.array([[x[1],x[0],0], [0,x[2],x[1]],[1,-1,0]])
    x0 = np.array([0.5,0.5,0.5])
    print(newton(f,x0,Df,alpha=0.3,maxiter=200))"""
    
    """#test problem 7:
    f = lambda x: (x**3)-1
    Df = lambda x: 3*(x**2)
    plot_basins(f, Df, [1, -0.5 + np.sqrt(3)/2 *1j, -0.5 - np.sqrt(3)/2 * 1j], domain=[-3/2, 3/2, -3/2, 3/2])
    
    g = lambda x: (x**3) - x
    Dg = lambda x: 3*(x**2) - 1
    plot_basins(g, Dg, [0, -1, 1], [-3/2, 3/2, -3/2, 3/2])"""
    
pass
