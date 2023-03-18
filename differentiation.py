# differentiation.py
"""Volume 1: Differentiation.
Jane Slagle
Math 347
1/16/22
"""

import sympy as sy
from matplotlib import pyplot as plt
import numpy as np
from autograd import elementwise_grad
from autograd import grad
from autograd import numpy as anp
import random
import time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    #define function f
    x = sy.symbols('x')
    f = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    f_func = sy.lambdify(x, f)
    
    deriv_f = sy.diff(f, x)     #take derivative with respect to x
    lamb_f = sy.lambdify(x, deriv_f)  #lambdify resulting function
    
    #test whether f and its derivative have the correct graph:
    """domain = np.linspace((-1)*np.pi, np.pi, 1000)
    plt.plot(domain, f_func(domain), color = "deeppink", label = "f")
    plt.plot(domain, lamb_f(domain), color = "indigo", label = "f prime")
    plt.legend(loc = "best")
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.show()"""
    
    return lamb_f

# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    
    return (f(x+h) - f(x)) / h
   
def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    
    return ((-3)*f(x) + 4*f(x+h) - f(x+(2*h))) / (2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    
    return (f(x) - f(x-h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    
    return (3*f(x) - 4*f(x-h) + f(x- (2*h))) / (2*h)  

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    
    return (f(x+h) - f(x-h)) / (2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    
    return (f(x-(2*h)) - 8*f(x-h) + 8*f(x+h) - f(x+(2*h))) / (12*h)

# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    #make the h's need, using np.logspace will put them in an array:
    h_vals = np.logspace(-8, 0, 9)
    
    #use prob 1 to find exact value of f'(xo):
    f_prime = prob1()    #first get the derivative of f by calling prob 1
    f_prime_val = f_prime(x0)
    
    #need f function so that can plug it in prob 2 functions:
    x = sy.symbols('x')
    f = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    f = sy.lambdify(x, f, "numpy")
    
    #lists to keep track of errors for each prob 2 function:
    fw1 = []
    fw2 = []
    bw1 = []
    bw2 = []
    ce1 = []
    ce2 = []
    
    #track the absolute error:
    for h_i in h_vals:
        #loop through all h values and append the absolute error for each trial for all h values
        fw1.append(np.abs(f_prime_val - fdq1(f, x0, h_i)))
        fw2.append(np.abs(f_prime_val - fdq2(f, x0, h_i)))
        bw1.append(np.abs(f_prime_val - bdq1(f, x0, h_i)))
        bw2.append(np.abs(f_prime_val - bdq2(f, x0, h_i)))
        ce1.append(np.abs(f_prime_val - cdq2(f, x0, h_i)))
        ce2.append(np.abs(f_prime_val - cdq4(f, x0, h_i)))
        
    #plot the absolute error against h on log-log scale (plot the lists against h):
    plt.loglog(h_vals, fw1, marker = '*', color = "mediumvioletred", label = "Order 1 Forward")
    plt.loglog(h_vals, fw2, marker = '*', color = "darkcyan", label = "Order 2 Forward")
    plt.loglog(h_vals, bw1, marker = '*', color = "deepskyblue", label = "Order 1 Backward")
    plt.loglog(h_vals, bw2, marker = '*', color = "orangered", label = "Order 2 Backward")
    plt.loglog(h_vals, ce1, marker = '*', color = "darkviolet", label = "Order 2 Centered")
    plt.loglog(h_vals, ce2, marker = '*', color = "darkorange", label = "Order 4 Centered")
    
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.legend(loc = "best")
    plt.show()

# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    #get the data from the plane.npy file:
    airplane = np.load("plane.npy")
    t = airplane[:, 0]              #first column are times
    alpha = airplane[:, 1]          #2nd column is alpha
    beta = airplane[:, 2]           #3rd column is beta
    
    a = 500
    h = 1
    speedy_mcspeedster = []
    
    #want first oder fward deriv of x, y, so need x_0+h and x_0 values (x_0+h is the next one):
    #need alpha0, beta0 to get the x_0 , y_0 position
    for i in range(len(airplane[0:,0])):  #loop through all data in plane
        if i == 0:   #t=7 is first one, so when i is 0
            #convert alpha and beta to radians:
            alpha0 = np.deg2rad(airplane[i][1])
            beta0 = np.deg2rad(airplane[i][2])
            alpha1 = np.deg2rad(airplane[i+1][1])  #get alpha, beta for x0 + h part and know h is 1 here
            beta1 = np.deg2rad(airplane[i+1][2])
        
            x0 = a * (np.tan(beta0) / (np.tan(beta0) - np.tan(alpha0)))
            y0 = a * (np.tan(beta0)*np.tan(alpha0) / (np.tan(beta0) - np.tan(alpha0)))
            x1 = a * (np.tan(beta1) / (np.tan(beta1) - np.tan(alpha1)))                 #x0+h
            y1 = a * (np.tan(beta1)*np.tan(alpha1) / (np.tan(beta1) - np.tan(alpha1)))
    
            #want first order forward deriv. forward 1 in table: get x', y'
            x_prime = x1 - x0
            y_prime = y1 - y0
            
            #find the speeds:
            speedy_speedster = np.sqrt((x_prime)**2 + (y_prime)**2)
            speedy_mcspeedster.append(speedy_speedster)
            
        elif i == 7:   #t=14 is last one
            alpha0 = np.deg2rad(airplane[i-1][1])
            beta0 = np.deg2rad(airplane[i-1][2])
            alpha1 = np.deg2rad(airplane[i][1])  #get alpha, beta for x0 + h part
            beta1 = np.deg2rad(airplane[i][2])
        
            x0 = a * (np.tan(beta0) / (np.tan(beta0) - np.tan(alpha0)))
            y0 = a * (np.tan(beta0)*np.tan(alpha0) / (np.tan(beta0) - np.tan(alpha0)))
            x1 = a * (np.tan(beta1) / (np.tan(beta1) - np.tan(alpha1)))                      #x0+h
            y1 = a * (np.tan(beta1)*np.tan(alpha1) / (np.tan(beta1) - np.tan(alpha1)))
            
            #use first order backward difference deriv: backward 1 in table
            x_prime = x1 - x0
            y_prime = y1 - y0
            
            #find the speeds:
            speedy_speedster = np.sqrt((x_prime)**2 + (y_prime)**2)
            speedy_mcspeedster.append(speedy_speedster)
            
        else:
            alpha0 = np.deg2rad(airplane[i-1][1])
            beta0 = np.deg2rad(airplane[i-1][2])
            alpha2 = np.deg2rad(airplane[i+1][1])  #get alpha, beta for x0 + h part
            beta2 = np.deg2rad(airplane[i+1][2])
        
            x0 = a * (np.tan(beta0) / (np.tan(beta0) - np.tan(alpha0)))
            y0 = a * (np.tan(beta0)*np.tan(alpha0) / (np.tan(beta0) - np.tan(alpha0)))
            x2 = a * (np.tan(beta2) / (np.tan(beta2) - np.tan(alpha2)))                      #x0+h
            y2 = a * (np.tan(beta2)*np.tan(alpha2) / (np.tan(beta2) - np.tan(alpha2)))
            
            #use second order centered difference deriv for everything in the middle:  2 in the table
            x_prime = (x2 - x0) / (2*h)
            y_prime = (y2 - y0) / (2*h)
            
            #find the speeds:
            speedy_speedster = np.sqrt((x_prime)**2 + (y_prime)**2)
            speedy_mcspeedster.append(speedy_speedster)
            
    return np.array(speedy_mcspeedster)
            
# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    #approx matrix by using equation 8.5 and equation 8.5 gives you each entry in the matrix (shows this in lab manual)
    #get jth standard basis vector (jth column of nxn identity matrix)
    n = len(x)
    ident = np.identity(n)
    approx_jacob = []       #list to store all the derivatives from equation 8.5 in (8.5 gives you each derivative need)
    
    for e_j in ident:  #for loop loops through each row of identity matrix (because jth row, jth column of identity matrix are same)
        deriv = (f(x + h*e_j) - f(x - h*e_j)) / (2*h)
        approx_jacob.append(deriv)
    jacob_approx = np.array(approx_jacob).T     #convert the list to an array and take transpose. the derivatives are the columns of jacobian
    
    return jacob_approx

# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    #formula from book to find nth Chebyshev polynomial at x
    return anp.cos(n*anp.arccos(x))   #equation 9.21 page 421 in book
    
def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    #find derivative of Tn(x):
    x = np.linspace(-1, 1, 1000)
    n_vals = [0, 1, 2, 3, 4]
    
    for n in n_vals:
        T_prime = elementwise_grad(cheb_poly)  #creates new lambda function for the derivative of T (since lambda function it takes the
                                               #same input parameters as cheb_poly does
        plt.subplot(2, 3, n+1)                 #plot them all on subplots
        plt.plot(x, T_prime(x, n), color = "deeppink", label = "n = " + str(n))     #plot the derivative over domain x and at each n value
        plt.legend(loc = "best")
    plt.show()

# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    #use lambda function for f instead of sympy function because otherwise the grad function won't 
    #work because sy.sin doesnt work
    f = lambda x: (anp.sin(x) + 1)**(anp.sin(anp.cos(x)))
     
    prob_1_times = []      #list to store times for calling prob 1
    cdq4_time = []   #list to store the CDQ4 times in
    grad_times = []  #list to store the autograd grad times in
    prob1_error = [1e-18]*200   #told to hardcode this error
    cdq4_error = []
    grad_error = []
    
    #want to do everything N times so need to loop through N:
    for n in range(N):
        x_0 = np.random.random()     #choose random value x_0
        
        #time how long it takes to call function:
        start1 = time.time()
        f_prime = prob1()(x_0)       #call problem 1 and plug x_0 into it
        end1 = time.time() - start1
        prob_1_times.append(end1)
        
        #time how long 4th order centered difference quotient from prob 3 takes:
        start2 = time.time()
        CDQ4 = cdq4(f, x_0)
        end2 = time.time() - start2
        cdq4_time.append(end2)
        #get absolute error:
        step3_error = np.abs(CDQ4 - f_prime)
        cdq4_error.append(step3_error)
        
        #time how long it takes to get an approx using Autograd
        start3 = time.time()
        auto_grady = grad(f)(x_0)
        end3 = time.time() - start3
        grad_times.append(end3)
        #get the absolute error:
        step4_error = np.abs(f_prime - auto_grady)
        grad_error.append(step4_error)
        
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(prob_1_times, prob1_error, color = "mediumvioletred", label = "SymPy", alpha = 0.5)
    plt.scatter(cdq4_time, cdq4_error, color = "mediumpurple", label = "Difference Quotients", alpha = 0.5)
    plt.scatter(grad_times, grad_error, color = "lightskyblue", label = "Autograd", alpha = 0.5)
    plt.legend(loc = "best")
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel("Absolute Error")
    plt.show()
        

