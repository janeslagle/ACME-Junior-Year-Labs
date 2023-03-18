# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
Jane Slagle
Math 323
1/27/22
"""

import numpy as np
from scipy import optimize as opt
from autograd import numpy as anp
from autograd import grad
from matplotlib import pyplot as plt

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #follow algorithm 11.1 in lab manual:
    x0 = (a+b)/2   #set initial min approx as interval midpoint
    phi = (1+np.sqrt(5))/2
    converges = False          #bool to return if converges or not
    counter = 0
    for i in range(maxiter):    #iterate only maxiter times at most
        counter += 1            #want to return the number of iterations computed
        c = (b-a)/phi
        a_bar = b-c
        b_bar = a+c
        
        if f(a_bar) <= f(b_bar):   #get new boundaries for search interval
            b = b_bar
        else:
            a = a_bar
        x1 = (a+b)/2           #set minimizer approx as interval midpoint
        if np.abs(x0-x1) < tol:
            converges = True   #if enter this if statement: then it converges
            break              #stop iterating if the approx stops changing enough
        x0 = x1                #x1 will be the approx min
         
    return x1, converges, counter   

# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #use equation 11.1 in lab manual to get local optimizer via Newton's method:
    #base answer off of what did in newtons method lab for volume 1:
    counter = 0        #want return number of iterations computed
    converges = False  #want return whether or not algorithm converged
    
    for k in range(maxiter):   #want iterate until reach max # iteration times
            counter += 1 
            x1 = x0 - (df(x0)/d2f(x0))  #equation 11.1 in lab manual 
            if np.abs(x1 - x0) < tol:   #check if converges, want stop iterating if does
                converges = True
                break 
            x0 = x1                     #update xk to be x_k+1 for the next iteration. x1 is the local optimizer want
            
    return x1, converges, counter
   
# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    counter = 0
    converges = False
    
    for k in range(maxiter):
        counter += 1
        xk_deriv = df(x1)       #told in lab manual to only find the derivative once for each k
        other_deriv = df(x0)   
        x2 = ((x0*xk_deriv) - (x1*other_deriv)) / (xk_deriv - other_deriv)  #equation 11.3 in lab manual. x0 = x_k-1, x1= x_k, and new x2= the new x_k+1
        if np.abs(x2 - x1) < tol:   #check if converges, want stop iterating if does
                converges = True
                break 
        x0 = x1                     #update x values for next time through the iteration
        x1 = x2
            
    return x2, converges, counter
    
# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    #do algorithm 11.2 from lab manual:
    #alpha is the step size here
    
    Dfp = np.dot(Df((x).T),(p))         #compute these values only once
    fx = f(x)
    
    while (f(x+(alpha*p)) > fx + (c*alpha*Dfp)):
        alpha = rho*alpha
    
    return alpha
    
if __name__ == '__main__':
    """#test problem 1:
    f = lambda x: np.exp(x) - 4*x
    print(golden_section(f, 0, 3))"""
    
    """#test prob 2:
    df = lambda x: 2*x + 5*np.cos(5*x)
    d2f = lambda x: 2 - 25*np.sin(5*x)
    print(newton1d(df, d2f, 0, 1e-10, 500))"""
    
    """#test prob 3:
    df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    func_answer = secant1d(df, 0, -1, 1e-10, 500)
    print(func_answer)
    print(opt.newton(df, x0=0, tol=1e-10, maxiter = 500))
    domain = np.linspace(-2, 2, 1000)
    f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    plt.plot(domain, f(domain), color = "deeppink")
    plt.plot(func_answer[0], f(func_answer[0]), '.', color = "navy", markersize = 10)
    plt.show()"""
    
    """#test prob 4:
    f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
    x = anp.array([150., .03, 40.])
    p = anp.array([-.5, -100., -4.5])
    print(backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4))"""
    
pass
