# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Jane Slagle
Math 323
1/12/22
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from numpy.fft import fft

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    #points is like domain, so x in 9.1 are from points
    #compute denominator of each L_j (using equation 9.1): both x_j and x_k are from xint
    #want to evalulate L_j (the polynomial at all points in domain) so need to find the numerator next and can do it in same
    #for loop
    L = []  #list to store the denom/numerator values at
    
    for j in xint:
        denom = 1
        numerator = 1
        for k in xint:
            if not np.allclose(k, j):
                denom *= (j - k)
                numerator *= (points - k)   #can do this and will array broadcast each time
        L.append(numerator/denom)      #step number 3: need to get the values at each, so need to find num/denom
     
    results = np.array(L)                #turn the results into an array
    
    #problem 2:
    #do equation 9.2:
    p = 0                          #p will be the polynomial
    for j in range(0, len(xint)):  #xint has length n
        p += (yint[j] * results[j])   #get polynomial p in equation 9.2. p is an array already so dont need to turn into array
        
    return p
    
# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.xint = xint
        self.yint = yint
        
        #find corresponding barycentric weights: use formula in lab manual to find them
        n = len(xint)    #get number of interpolating points
        w = []           #list to store the weights in
        
        C = (np.max(xint) - np.min(xint)) / 4   #find the capacity C
        
        for j in xint:
            prod = 1
            for k in xint:
                if k != j:
                    prod *= ((j - k) / C)
                    weight = 1 / prod
            w.append(weight)
            
        weights = np.array(w)   #turn the list of weights into array
        
        self.weights = weights
        
    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        #find p(x) using formula given in lab manual
        x = self.xint
        y = self.yint
        w = self.weights
        
        poly = []
        numerator = 0
        denominator = 0
        for j in range(0, len(x)):
            numerator += (w[j] * y[j]) / (points - x[j])
            denominator += ((w[j]) / (points - x[j])) 
        p = numerator / denominator
        poly.append(p)
            
        return np.sum(poly, axis = 0)  
        
    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        xints = np.concatenate((self.xint, xint))  #extend class attribute arrays that store x values
        indices = np.argsort(xints)                #indices to sort the x's in order since added new x values to it
        self.xint = xints[indices]                 #this actually sorts the x values
       
        self.yint = np.concatenate((self.yint, yint))[indices]  #extend class attribute arrays that store y values. want sort y
                                                            #values according to the x values
       
        self.__init__(self.xint, self.yint)    #call the constructor with the extended x, y values since we compute the weights
                                               #in the constructor
       
# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #get domain of 400 equally spaced pts on [-1,1]
    domain = np.linspace(-1, 1, 400)
    
    #function to be interpolated
    f = lambda x: 1/(1+25 * x**2)
    
    n_vals = [2**i for i in np.arange(2,9)]      #n values want to loop over
    errors_1 = []                                   #list to store all the errors because plotting them
    errors_2 = []                 
    
    for n in n_vals:
        points = np.linspace(-1, 1, n)           #want to interpolate with n equally spaced points over [-1, 1]
        cheb_points = np.array([np.cos(j * np.pi/n) for j in range(n+1)])  #get the chebyshev extremizer points (just formula)
                                                                           #get the points for part 2
        
        poly = BarycentricInterpolator(points)   #polynomail object that interpolates through the points. BarycentricInterpolator
                                                 #is a class, so this is creating the class object with points plugged in
        cheb_poly = BarycentricInterpolator(cheb_points)   #get class object for the chebyshev points
        yint = f(points)
        poly.set_yi(yint)                        #get corresponding y values to x values because we need them to interpolate.
                                                 #can't just interpolate a function with only x values (because formula literally
                                                 #requires y values)
        cheb_poly.set_yi(f(cheb_points))         #set the y values for the chebyshev stuff                                         
        
        error_1 = np.linalg.norm(f(domain) - poly(domain), ord = np.inf)  #find absolute error with infinity norm
        errors_1.append(error_1)
        error_2 = np.linalg.norm(f(domain) - cheb_poly(domain), ord = np.inf)
        errors_2.append(error_2)
    
    #plot the errors against number of interpolating points n in loglog plot
    plt.loglog(n_vals, errors_1, color = "mediumvioletred", label = "Uniformly Spaced Interpolating Error")
    plt.loglog(n_vals, errors_2, color = "plum", label = "Chebyshev Points Interploating Error")
    
    plt.legend(loc = "best")
    plt.title("Errors against n values")
    plt.show()
       
# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    y = np.cos((np.pi * np.arange(2*n)) / n)  #get the chebyshev extremizers
    samples = f(y)                            #plug the extreme points into the function
    
    coeffs = np.real(fft(samples))[:n+1] / n   
    coeffs[0] = coeffs[0]/2                    
    coeffs[n] = coeffs[n]/2
    
    return coeffs
    
# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    #load the data in the file
    data = np.load("airdata.npy")
    
    #copy and paste the whole entire yellow box:
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = Barycentric(domain[temp2], data[temp2])    #our own Barycentric function. This is the interpolating polynomial
    
    #now plot the polynomial on 2 separate subplots:
    plt.subplot(2, 2, 1) 
    plt.plot(domain, poly(domain), color = "deeppink")
    
    plt.subplot(2, 2, 2)
    plt.scatter(domain, data, s = 2, color = "violet")
    
    plt.suptitle("Polynomial interpolation of Air Quality Data", color = "darkblue")
    plt.show()

if __name__ == '__main__':
    """#test problems 1, 2:
    domain = np.linspace(-1, 1, 1000)
    f = lambda x: 1 / (1+ 25*(x**2))
    
    #plot the stuff from probs 1, 2: base it off of figure 9.1
    #the points variable when plug into lagrange is the same as domain
    xs = np.linspace(-1, 1, 5)     #want 5 equally spaced interpolating points
    ys = f(xs)                     #get y values by plugging x values into f function
    
    plt.plot(domain, f(domain), color = "mediumvioletred", label = "original")
    plt.plot(domain, lagrange(xs, ys, domain), color = "indigo", label = "interpolation")
    plt.legend(loc = "best")
    plt.show()"""
    
    """#test problem 3:
    domain = np.linspace(-1, 1, 1000)
    f = lambda x: 1 / (1+ 25*(x**2))
    xs = np.linspace(-1, 1, 5)
    ys = f(xs)
    
    b = Barycentric(xs, ys)  
    
    plt.plot(domain, f(domain), color = "darkslategray")
    plt.plot(domain, b(domain), color = "deeppink")
    plt.show()"""
    
    """n = 11
    runge = lambda x: 1 / ( 1 + 25 * x ** 2)
    x = np.linspace(-1, 1, n)
    y = runge(x)
    domain = np.linspace(-1, 1, 1000)
    b = Barycentric(x, y)
    output = b(domain)
    plt.plot(domain, runge(domain), 'c-', label='Original')
    plt.plot(domain, output, 'r-', label='Interpolation')
    plt.legend(loc='best')
    plt.show()"""
    
    """n = 11
    runge = lambda x: 1 / (1 + 25 * x**2)
    xvals_original = np.linspace(-1, 1, n)
    xvals_1 = xvals_original[1::2]
    xvals_2 = xvals_original[::2]
    domain = np.linspace(-1, 1, 1000)
    bary = Barycentric(xvals_1, runge(xvals_1))
    
    bary_2 = Barycentric(xvals_original, runge(xvals_original))
    plt.plot(domain, bary_2(domain),linewidth=6, label='Not added')
    plt.plot(domain, runge(domain), label='Original')
    plt.plot(domain, bary(domain), label='Odd Points, n = ' + str(n))
    bary.add_weights(xvals_2, runge(xvals_2))
    plt.plot(domain, bary(domain),'k', label='All points, n = ' + str(n))
    plt.legend(loc='best')
    plt.show()"""
   
    pass
    
