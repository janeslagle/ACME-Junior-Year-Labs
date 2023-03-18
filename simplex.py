"""Volume 2: Simplex
Jane Slagle
2/22/24
Vol 2 lab
"""

import numpy as np

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        x = np.zeros(len(c))       #need create x as array of 0s
        
        if not all(np.greater(b, A@x)):     #np.greater does coordinate wise inequality: doing greater checks if not feasible
            raise ValueError("The given system is infeasible at the origin")    #raise value error if not feasible
            
        self.c = c                 #save everything as attributes
        self.A = A
        self.b = b
        
        #call problem 2:
        self._generatedictionary(c, A, b) 
        
    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        Im = np.identity(len(b))     #need this to make A_bar 
        A_bar = np.hstack((A, Im))   #use hstack bc stacks arrays column wise and want [A Im] 
        
        ZERO = np.zeros(len(b))      #need this to make c_bar. want last m entries to be zeros
        c_bar = np.concatenate((c, ZERO))  #np.conctenate will add ZERO as the next row in the array like want
        
        #make initial dictionary:
        zero_b = np.concatenate((np.array([0]), b))     #makes row with 0 and then b
        first_col = np.reshape(zero_b, (len(b)+1, 1))   #need reshape zero_b to be column bc originally, it is a row
                                                        #right now it has shape (1, len(b) + 1), but want to be (len(b) + 1, 1)
        sec_col = np.vstack((c_bar.T, -1*A_bar))      
        init_dict = np.hstack((first_col, sec_col))    #hstack then stacks them as 2 columns
        
        #save dictionary as attribute:
        self.D = init_dict
       
    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        #this part is like getting the leaving variable
        
        for i, entry in enumerate(self.D[0,1:]):   #loop through all entries in 1st row of dictionary. do enumerate bc that will give the entry and
                                           #the index and want to return the INDEX of the first negative entry. enumerate will allow to 
                                           #check the entry and get the index of that entry at the same time
            if entry < 0:
                break
        self.col_ind = i+1                  #i will be the index of the first negative entry in first row of dict
        
        return self.col_ind
               
    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        #if all entries in the pivot column are nonnegative: problem is unbounded, has no solution. have aviod this so raise value error
        if np.all(self.D[1:, index] >= 0):
            raise ValueError("it gives you an unbounded solution!!!!")
            
        #need to find the ratios to use Bland's Rule (since doing Bland's Rule here):
        ratios = []         #store the ratios, indices in lists
        indices = []
        
        for i in range(1, len(self.D[:, 0])):    #loop through all rows
            #make sure don't divide by 0, so only find the ratio if it's not 0
            if self.D[i][index] != 0:            #check to make sure the entry isnt 0
                #find the ratio given in lab manual:
                hoodrat = -self.D[i][0] / self.D[i][index]   #index is j here
                
                if hoodrat > 0:     #only want to append ratio if it's positive
                    ratios.append(hoodrat)
                    indices.append(i)
        
        return indices[np.argmin(ratios)]    #want to return the smallest index out of all the ratios
            
    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        #first get the column, row to pivot on
        col_ind = self._pivot_col()
        row_ind = self._pivot_row(col_ind)
        
        #now do row operations on the dictionary as described in lab manual:
        #divide pivot row by negative value of pivot entry
        self.D[row_ind] /= (-1*self.D[row_ind][col_ind])    #need set row in dictionary equal to this
        
        #get pivot entry:
        entry = self.D[row_ind][col_ind]
        
        #use pivot row to zero out all entries in pivot column above, below pivot entry:
        for i in range(len(self.D)):   #loop through all rows of dictionary
            if i != row_ind:    #get everything except the row of pivot entry
                self.D[i] -= (self.D[i][col_ind]/entry)*self.D[row_ind]    #zero it out using normal row operations
                #doing this just updates the dictionary automatically like want

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        #first check if done solving (if the optimal point has been found):
        while not np.all(self.D[0, 1:] >= 0):  #this checks if each entry in 1st row of dict is greater than 0 
            self.pivot()      #pivot until get that ALL entires in first row of dict are nonneg
        
        #get the min value which is upper leftmost entry of dict
        min_val = self.D[0][0]  
        
        dep_var = {}      #in each dict: the index of variable is key, value is the entry in OG dict at that index
        indep_var = {}
        
        for i, entry in enumerate(self.D[0,1:]):   #loop through the first row of the dict, skipping 1st entry
            if entry == 0:                         #get dep variables
                for j in range(len(self.D)):  #loop through all rows in dict
                    if self.D[j][i+1] == -1:    #for column that the 0 entry is in: go through all rows in that column and see which entry = -1
                        dep_var[i] = self.D[j, 0]    #make i-1 a key in dict with corresp value entry in 1st col of dict in that row
            
            else:                        #get indep variables
                indep_var[i] = 0                 #indep variables will have value 0 in dict
    
        return min_val, dep_var, indep_var
            
# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    data = np.load(filename)
    
    #get A, p, m, d arrays from data:
    A = data['A']    #resource coeff (are 4 products x 3 resources)
    p = data['p']    #gives unti prices of 4 products
    m = data['m']    #gives available resource untis
    d = data['d']    #gives demand constraints for each of 4 products
    
    #want use SimplexSolver class on this info from file, so need take A,p,m,d and get c,A,b to plug into the class
    c = -1*p                    #told that need to make it a min problem, so need multiply the p's by -1 (p is c here since its obj func)
    
    _, n = A.shape              #need to get the dimensions of A because need to stack A with the identity matrix
    A_want = np.row_stack((A, np.identity(n)))
    b = np.concatenate((m, d))  #get all the constraints: m and d are both constraints have
    
    S = SimplexSolver(c, A_want, b)   #make class object so that can call solve method on it
    
    dep = S.solve()[1]
    indep = S.solve()[2]
    num_units = []
    
    for key in dep:   #loop through all they keys in the dict
        if key < 4:   #so loops through keys from 0 to 3 (want it to loop through 0 to n-1)
            num_units.append(dep[key])  #add the value from dict at that key
    for key in indep:
        if key < 4:
            num_units.append(indep[key])
    
    return np.array(num_units)          #convert the list to an array   
    
#function to test everything:    
def testing():
    S = SimplexSolver(np.array([-3,-2]), np.array([[1,-1,],[3,1],[4,3]]), np.array([2,5,7]))
    
    #print(S.D)                               #for testing prob 2
    #print(S._pivot_row(S._pivot_col()))      #for testing prob 3
    
    """#test prob 4:
    S.pivot()     #update dictionary by calling pivot
    print(S.D)    #now print the updated dictionary"""
    
    print(S.solve())     #test prob 5
    
   
