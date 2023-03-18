# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Jane Slagle
Volume 2 Lab
3/9/22
"""

import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #first make the objective function:
    x = cp.Variable(3, nonneg = True)        #x variable needs be a CVXPY variable with 3 components
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)
    
    #write constraints:  have no A constraint bc have no equality constraints
    G = np.array([[1, 2, 0], [0, 1, -4]])  #G is coeffs of x in <= inequality constraints
    h = np.array([3, 1])                   #h is RHS of the <= inequality constraints
    P = np.array([[2, 10, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  #P is for all >= inequalities
    q = np.array([12, 0, 0, 0])            #q is RHS of the >= inequal constraints
    constraints = [G @ x <= h, P @ x >= q] #constraints must be list
    
    #assemble problem:
    problem = cp.Problem(objective, constraints)
    opt_value = problem.solve()            #need solve problem before get x.value
    optimizer = x.value
    
    return optimizer, opt_value
    
# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(len(A[0]))   #the number of x's corresponds to number of columns in matrix A
    objective = cp.Minimize(cp.norm(x, 1))
    
    #get constraints:
    constraints = [A @ x == b]
    
    problem = cp.Problem(objective, constraints)
    opt_value = problem.solve()            #need solve problem before get x.value
    
    return x.value, opt_value
    
# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(6, nonneg = True)   #number of x's is the number of variables have: have p1-p6, so x needs be 6
    c = np.array([4, 7, 6, 8, 8, 9])    #shipping costs for each piano have
    objective = cp.Minimize(c.T @ x)    #get the objective function
    
    #get everything need for constraints:
    #make matrix A that corresponds to coefficients of all variables in each of the constraint equations have:
    A = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0 ,0], [0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    constraints = [A @ x == np.array([7, 2, 4, 5, 8])]   #create the actual constraints now
    
    problem = cp.Problem(objective, constraints)
    opt_value = problem.solve()            #need solve problem before get x.value
    
    return x.value, opt_value

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #need write g in the quadratic problem form:
    #x will be 3x1 vector, so Q will be 3x3 matrix, r will be 3x1 vector
   
    #when multiply out the quadratic objective function with form x, Q, r will be get:
    Q = np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]])
    r = np.array([3, 0, 1])
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))
    
    opt = prob.solve()
    opt_val = x.value
    
    return opt_val, opt
    
# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(len(A[0]), nonneg = True)   #the number of x's corresponds to number of columns in matrix A
    objective = cp.Minimize(cp.norm(A@x - b, 2))
    
    #get constraints:
    #equality must be affine constraint, but 1-norm is convex, so have write 1 norm with affine function
    #1 norm is the abs value sum of each component, but all var here are >= 0, so dont need the abs value part
    #just need the sum part. and then the rest of the problem is like normal
    constraints = [x @ np.ones(len(A[0])) == 1]   #have column vector of x's times row vector of 1s, so multiplying the 2 will sum all x's
    
    problem = cp.Problem(objective, constraints)
    opt_value = problem.solve()            #need solve problem before get x.value
    
    return x.value, opt_value

# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    food_table = np.load("food.npy", allow_pickle = True)   #data from file is the table described in lab manual
    
    #need to set up problem given:
    x = cp.Variable(18, nonneg = True)   #number of x's corresponds to number of columns in food table from data, and there are 18 cols
    objective = cp.Minimize(food_table[:, 0].T @ x)  #want minimize all of prices in 1st column of table times each x that we have
    
    #now get all of constraints from the food data table:
    #before can write the actual constraints, need 1st get all of the components in each constraint equation:
    #need multiply each of them by the number of servings, which is the 2nd column in data table
    calories = (food_table[:, 2]*food_table[:,1]).T @ x
    fat = (food_table[:, 3]*food_table[:,1]).T @ x
    sugar = (food_table[:, 4]*food_table[:,1]).T @ x
    calcium = (food_table[:, 5]*food_table[:,1]).T @ x
    fiber = (food_table[:, 6]*food_table[:,1]).T @ x
    protein = (food_table[:, 7]*food_table[:,1]).T @ x
    
    #now make the actual constraints:
    constraints = [calories <= 2000, fat <= 65, sugar <= 50, calcium >= 1000, fiber >= 25, protein >= 46] 
    
    #now solve the problem:
    problem = cp.Problem(objective, constraints)
    opt_value = problem.solve()            #need solve problem before get x.value
    
    return x.value, opt_value
   
#function to test everything    
def testing():
    """#test problem 2:
    A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
    b = np.array([7, 4])
    
    print(l1Min(A, b))"""
    
    """#test problem 5:
    A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
    b = np.array([7, 4])
    
    print(prob5(A, b))"""

