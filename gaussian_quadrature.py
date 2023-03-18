# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
Jane Slagle
Math 323
1/20/22
"""

import scipy.linalg as la
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from matplotlib import pyplot as plt

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        if polytype != "legendre" and polytype != "chebyshev":
            raise ValueError("The label is not either legendre or chebyshev")
        
        self.polytype = polytype     #save the label as an attribute
        self.n = n
        
        #get what w(x) equals:
        if polytype == "legendre":   #if legendre then w is just 1
            w = lambda x: 1
        else:                        #if chebyshev then w is given by formula in lab manual
            w = lambda x: 1 / (np.sqrt(1-(x**2)))
            
        #define w inverse function with lambda function
        winv = lambda x: 1 / w(x)
        
        self.w_inverse = winv        #save w inverse as an attribute
        
        #save prob 2 stuff as attributes:
        prob_2 = self.points_weights(n)
        self.points = prob_2[0]     #the points are first thing returned in prob 2
        self.weights = prob_2[1]    #the weights are 2nd thing returned in prob 2

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        #construct Jacobian matrix J for polynomial family indicated in constructor, so first get the label from prob 1
        poly_fam = self.polytype
        
        a = [0]*(n)             #initialize a, b. do n-1 because want a, b from 1 to n
        b = [1]*(n-1)           #the b's are only on the offset diagonals, so will always have one less than n           
        
        #get a, b values for legnedre:
        if poly_fam == "legendre":
            mew = 2
            for k in range(len(b)):    #want to loop through n times
                b[k] = (((k+1)**2) / (4*((k+1)**2) - 1))
        #get a, b values for chebyshev:
        if poly_fam == "chebyshev":
            mew = np.pi
            for k in range(0, n-1):
                if k == 0:             #want 1st b value to be 1/2
                    b[k] = (1/2)
                else:
                    b[k] = (1/4)
                    
        
        #construct the Jacobian matrix:
        upper_diag = np.diag(np.sqrt(b), k = 1)    #get the diagonal above the main diagonal: the k says which diagonal in matrix want put b in
        lower_diag = np.diag(np.sqrt(b), k = -1)   #get diagonal below main diagonal: the k specifies which diagonal in matrix want put b in 
        J = upper_diag + lower_diag
        
        #get eigenvalues, eigenvectors of J:
        x_vals = la.eigh(J)[0]        #eigh returns eigvals, then eigvecs. the x points are the eigvals
        eigvecs = la.eigh(J)[1]       #get the eigenvectors
        weights = [((eigvecs[0,i])**2)*mew for i in range(len(eigvecs))]  #get weights according to formula in lab manual
        
        return x_vals, weights

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        #get weights, points (the x values) that found in problem 2:
        pts = self.points
        weight = self.weights
        
        #need to get new function g: g(x) = f(x)/w(x)
        w_inverse = self.w_inverse     
        g = lambda x: f(x) * w_inverse(x)
        
        #now approximate integral using equation 10.1:
        int_approx = 0                 
        for i in range(self.n): 
            int_approx += (g(pts[i])*weight[i])
            
        return int_approx

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        #use equation 10.2 in lab manual to approx integral of f from a to b:
        #first define h because need it to define g: g = h/x
        h = lambda x: f(((b-a)/2)*x + ((a+b)/2))
        
        #need create class object so that can call problem 3
        n = self.n                          #need to get the 2 parameters needed to make class object from constructor
        this_dumb_thing = self.polytype
        class_object = GaussianQuadrature(n, this_dumb_thing)
        
        summation = class_object.basic(h)      #gets summation in equation 10.2       
        int_approx = ((b-a)/2) * summation     #from equation 10.2
        
        return int_approx

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        #use equation 10.5 in lab manual to find the double integral: 
        #need to get h and g functions:
        h = lambda x, y: f( ((b1-a1)/2)*x + ((a1+b1)/2), ((b2-a2)/2)*y + ((a2+b2)/2) )
        g = lambda x, y: h(x,y)*(self.w_inverse(x)*self.w_inverse(y))
        
        pts = self.points
        weights = self.weights
        
        sum_this_up = 0
        for i in range(self.n):
            for j in range(self.n):
                sum_this_up += weights[i]*weights[j]*g(pts[i],pts[j])
               
        dont_forget_me = ((b1-a1)*(b2-a2))/4
        int_approx = dont_forget_me*sum_this_up
        return int_approx
        
# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    #first find "exact" value F of integral given from -3 to 2:
    exact_int = norm.cdf(2) - norm.cdf(-3)      
    f = lambda x: (1 / (np.sqrt(2*np.pi)))*np.exp((-1*(x**2))/2)
    
    legend_errors = []    #list to store legendre approximation erros in
    chubby_errors = []   #list to store chebyshev approximation errors in
    
    for n in np.arange(5, 55, 5):
        woo = GaussianQuadrature(n, "legendre")      #need create class object with legendre polynomials
        legend_approx = woo.integrate(f, -3, 2)      #use class problem 4 to approximate the integral
        legend_error = np.abs(exact_int - legend_approx)
        legend_errors.append(legend_error)
        
        boo = GaussianQuadrature(n, "chebyshev")
        chubby_approx = boo.integrate(f, -3, 2)
        chubby_error = np.abs(exact_int - chubby_approx)
        chubby_errors.append(chubby_error)
    
    n = np.arange(5, 55, 5)
    scipy_error = [np.abs(exact_int - quad(f, -3, 2))[0]]*len(n)
    
    #plot errors against number of points, weights (which is n) using log scale for y-axis
    plt.yscale('log')          #want x axis to be log scale
    plt.plot(n, legend_errors, color = "deeppink", label = "Legendre")
    plt.plot(n, chubby_errors, color = "firebrick", label = "Chebyshev")
    plt.plot(n, scipy_error, color = "steelblue", label = "Scipy")
    
    plt.xlabel("Number of points and weights n")
    plt.ylabel("Errors of integral approximations")
    plt.legend(loc = "best")
    plt.show()

if __name__ == '__main__':
    """#to test problem 2: want it to return True, True
    test_points = np.array([(-1/3) * np.sqrt(5 + 2*np.sqrt(10/7)),
                             (-1/3) * np.sqrt(5 - 2*np.sqrt(10/7)),
                            0,
                             (1/3) * np.sqrt(5 - 2*np.sqrt(10/7)),
                             (1/3) * np.sqrt(5 + 2*np.sqrt(10/7))])

    test_weights = np.array([(322 - 13*np.sqrt(70)) / 900,
                             (322 + 13*np.sqrt(70)) / 900,
                             128 / 225,
                             (322 + 13*np.sqrt(70)) / 900,
                             (322 - 13*np.sqrt(70)) / 900])
    quad = GaussianQuadrature(5, 'legendre')

    print(np.allclose(quad.points, test_points))
    print(np.allclose(quad.weights, test_weights))
    """
    
    """#to test problem 6:
    f = lambda x, y: (x**2)-(y**2)
    blah = GaussianQuadrature(30, "legendre")
    print("first legendre: " + str(blah.integrate2d(f, -1, 1, -1, 1)))
    classy = GaussianQuadrature(30, "chebyshev")
    print("first chebyshev: " + str(classy.integrate2d(f, -1, 1, -1, 1)))
    
    print(" ")
    f = lambda x, y: (x**2)*(y**2)
    blah = GaussianQuadrature(30, "legendre")
    print("second legendre: " + str(blah.integrate2d(f, -1, 2, -3, 4)))
    classy = GaussianQuadrature(30, "chebyshev")
    print("second chebyshev: " + str(classy.integrate2d(f, -1, 2, -3, 4)))
    
    print(" ")
    f = lambda x, y: np.sin(x)*np.cos(y)
    blah = GaussianQuadrature(30, "legendre")
    print("thrid legendre: " + str(blah.integrate2d(f, -5, 6, -7, 8)))
    classy = GaussianQuadrature(30, "chebyshev")
    print("third chebyshev: " + str(classy.integrate2d(f, -5, 6, -7, 8)))"""
    
pass
