# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
Jane Slagle
Volume 2 Lab
2/16/22
"""

import numpy as np
from scipy import optimize as opt 
from  autograd import grad
from autograd import numpy as anp
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converges = False           #keep track on if converges or not
    counter = 0                 #store number of iterations computed
    
    for k in range(maxiter):    #want iterate until k > maxiter
        counter += 1
        #need get step size alpha_k: use equation in lab manual for exact steepest descent
        g = lambda alph: f(x0 - alph*Df(x0).T)          #get function to plug into argmin equation for alpha_k
        step_on_me = opt.minimize_scalar(g).x           #actually get alpha_k now
        x1 = x0 - (step_on_me * (Df(x0)).T)            
        if np.linalg.norm(Df(x0), ord = np.inf) < tol:   #want iterate until sup norm of Df(xk) is less than tol
            converges = True
            break
        x0 = x1                 #update for next loop through
    
    return x1, converges, counter

# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #follow algorithm 12.1 in lab manual:
    r0 = (Q @ x0) - b
    d0 = -r0
    counter = 0
    converges = False
    
    for k in range(len(b)):
        counter += 1
        alp_k = (r0.T @ r0) / (d0.T @ Q @ d0)         #this is a constant
        x1 = x0 + (alp_k * d0)    
        r1 = r0 + (alp_k * Q @ d0)
        beta1 = (r1.T @ r1) / (r0.T @ r0)             #this is a constant
        d1 = -r1 + (beta1 * d0)   
        if np.linalg.norm(r1, ord = np.inf) < tol:
            converges = True
            break
        x0 = x1         #update everything for next round through while loop
        r0 = r1
        d0 = d1

    return x1, converges, counter

# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    r0 = -df(x0).T
    d0 = r0
    #get alpha0:
    g = lambda alph: f(x0 + alph*d0)
    a0 = opt.minimize_scalar(g).x
    x0 = x0 + a0*d0
    counter = 1
    converges = False
    
    for k in range(maxiter - 1):
        counter += 1
        r1 = -1*(df(x0).T)
        betak = (r1.T @ r1) / (r0.T @ r0)    #rk = r1, rk-1 = r0
        d0 = r1 + betak*d0
        g = lambda alph: f(x0 + alph*d0)
        a0 = opt.minimize_scalar(g).x
        x0 = x0 + a0*d0
        r0 = r1
        
        if np.linalg.norm(r1, ord = np.inf) < tol:
            converges = True
            break
            
    return x0, converges, counter
        
# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    data = np.loadtxt(filename)   
    #first column of data (from file) will be the y values, which will be b, and then the rest of data from file will be in matrix A
    b = data[:, 0]     #says get all rows, 1st column
    A = data[:, 1:]    #matrix A will be columns of 1s, then all the rows, and all columns after 1st column from data file
    A = np.hstack((np.ones((len(b), 1)), A))   #add the column of 1s to A
    
    #want to solve equation 12.3 in lab manual, so need to plug in A.TA for Q, A.Tb for b in prob 2 function
    return conjugate_gradient(A.T@A, A.T@b, x0)[0]   #only want ot return the solution, so only return 1st thing from prob 2
    
# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        #get equation 12.4 in lab manual:
        f = lambda b: anp.sum([anp.log(1 + anp.exp(-1*(b[0] + b[1]*x[i]))) + (1-y[i])*(b[0] + b[1]*x[i]) for i in range(len(x))])
        df = grad(f)
        minnn = nonlinear_conjugate_gradient(f, df, guess)[0]
        
        #get the two attributes want:
        self.B0 = minnn[0]
        self.B1 = minnn[1]
        
    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        return 1 / (1 + np.exp(-1*(self.B0 + self.B1*x)))

# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    astronauts = np.load(filename)  #load the file
    
    #told that 1st column in file is temperature:
    temper = astronauts[:, 0]
    #told that 2nd column contains O-ring damage info:
    damaging = astronauts[:, 1]
    
    loggy = LogisticRegression1D()    #make class object
    throw_a_fit = loggy.fit(temper, damaging, guess)      #need to call fit on the data so that can use predicted on B0, B1 that get from it
    predicted = loggy.predict(np.linspace(30, 100, 1000))
    predicted31 = loggy.predict(31)
        
    domain = np.linspace(30, 100, 1000)
    plt.plot(domain, predicted, color = "darkmagenta")
    plt.scatter(31, predicted31, label = "P(Damage) at Launch", color = "deeppink")
    plt.scatter(temper, damaging, label = "Previous Damage", color = "mediumpurple")
    plt.title("Probability of O-Ring Damage")
    plt.legend(loc = "best")
    plt.xlabel("Temperature")
    plt.ylabel("O Ring Damage")
    plt.show()
    
    return predicted31
        
if __name__ == '__main__':
    """#test problem 1:
    f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    Df = grad(f)
    
    print(steepest_descent(f, Df, anp.array([1., 1., 1.]), tol=1e-5, maxiter=100))"""
    
    """#test prob 2:
    Q = np.array([[2,0],[0,4]])
    b = np.array([1,8]).T 
    x0 = np.array([0., 0.])
    print(conjugate_gradient(Q, b, x0))
    
    n = 4
    A = np.random.random((n,n))
    Q = A.T @ A
    b, x0 = np.random.random((2,n))
    x = conjugate_gradient(Q, b, x0)[0]
    print("prob 2 is: " + str(np.allclose(Q @ x, b)))"""
    
    """#test prob 3:
    print(opt.fmin_cg(opt.rosen, np.array([-2, 2]), fprime=opt.rosen_der))
    f = opt.rosen
    df = opt.rosen_der
    print(nonlinear_conjugate_gradient(f, df, np.array([-2, 2]), maxiter = 1000))"""

pass
