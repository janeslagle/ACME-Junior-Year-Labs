# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
Jane Slagle
Math 347
2/7/22
"""

import numpy as np
from scipy import linalg as la
from scipy import stats
from matplotlib import pyplot as plt

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    #draw N random points over n-dimensional domain [-1,1] x ... x [-1,1]:
    randos = np.random.uniform(-1, 1, (n, N))     #this is how they got their random points in lab manual ex
    
    #follow same process they use in lab manual to get the volume:
    #next 2 lines are for finding how many points are inside the unit ball
    lengths = la.norm(randos, axis = 0)
    num_within = np.count_nonzero(lengths < 1)
    
    return (2**n) * (num_within / N)              #told volume of domain = 2^n. return the estimated volume of Un
    
# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    #want sample N points over interval [a, b] using uniform distribution:
    samples = np.random.uniform(a, b, N)
    
    vol = b-a                 #told in lab manual that the volume is b-a here
    #use equation 11.2 to estimate integral:
    summ = 0
    for xi in samples: #want to sum from 1 to N
        summ += f(xi)    #sum up f(xi) values where each xi = from uniform distribution samples
    est_int = vol * (1/N) * summ #actually get equation 11.2 now
    
    return est_int          #return the integral estimate
   
# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    #mins variable is the a values
    #maxs variable is the b values
    n = len(mins)
    
    #equation 11.3: get the volume
    prod = 1
    for i in range(n):  #want to loop through n times
        prod *= (maxs[i] - mins[i])    #the volume
        
    #equation 11.4 stuff: gets the sample points need to plug into equation 11.1
    samps = np.random.uniform(0, 1, (N, n))    #want 1st sample uniformly from n-dim box [0,1] x ... x [0,1]
    for samp in samps:
        for i in range(n):
            samp[i] = (samp[i] * (maxs[i] - mins[i])) + mins[i]
    
    #equation 11.1: plug it all in
    int_est = prod * (1/N) * sum(f(xi) for xi in samps)
    
    return float(int_est)

# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    errors = []   
    n=4         
    #step 1:
    low = [-3/2, 0, 0, 0]
    high = [.75, 1, .5, 1]    #these are lower, upper bounds in omega given
    
    f = lambda x: (1 / ((2*np.pi)**(n/2))) * np.exp(-x.T @ x * .5)
    
    #step 2: 
    #define bounds of integration: make the mins and maxs into arrays
    mins = np.array(low)
    maxs = np.array(high)
    
    means, cov = np.zeros(n), np.eye(n)  
    sci_int = stats.mvn.mvnun(mins, maxs, means, cov)[0]   #compute integral with scipy
   
    #step 3:
    N_vals = np.logspace(1, 5, 20, dtype = int)  
    for N in N_vals:
        int_est = mc_integrate(f, mins, maxs, int(N))
        rel_err = np.abs(sci_int - int_est) / np.abs(sci_int)
        errors.append(rel_err)
    
    #step 4:
    plt.loglog(N_vals, errors, color = "deeppink", label = "Relative Error")
    plt.loglog(N_vals, 1 / np.sqrt(N_vals), color = "deepskyblue", label = "1/sqrt(N)")
    plt.legend(loc = "best")
    plt.show()
  
if __name__ == '__main__':
    """#test problem 1:
    print("should approx pi: " + str(ball_volume(2)))
    print("should be around 4.18879: " + str(ball_volume(3)))
    print("should be around 4.9348: " + str(ball_volume(4)))"""

    """"#test problem 2:
    f = lambda x: x**2
    print("real integral = 24: " + str(mc_integrate1d(f, -4, 2)))
    
    g = lambda x: np.sin(x)
    print("real integral = 0: " + str(mc_integrate1d(g, -2*np.pi, 2*np.pi, N = 100)))
    
    h = lambda x: 1/x
    print("real integral = about 2.30259: " + str(mc_integrate1d(h, 1, 10)))
    
    w = lambda x: np.abs(np.sin(10*x)*np.cos(10*x) + np.sqrt(x)*np.sin(3*x))
    print("real integral = about 4.502: " + str(mc_integrate1d(w, 1, 5)))"""
    
    """#test problem 3:
    f = lambda x: x[0]**2 + x[1]**2
    print("should be 2/3: " + str(mc_integrate(f, [0, 0], [1, 1], N=10000)))
    
    g = lambda x: 3*x[0] - 4*x[1] + x[1]**2
    print("should be 54: " + str(mc_integrate(g, [1, -2], [3, 1])))
    
    f0 = lambda x: x**2 
    print(mc_integrate(f0, mins = [-4], maxs = [2]))
    
    f3 = lambda x: np.sin(x[0]) - x[1]**3 
    print(mc_integrate(f3, [-1, -2], [2, 1]))"""
    
    pass
