# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
Jane Slagle
Math 321 Section 2
10/27/21
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath   #need for problem 6

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    
    Q_reduced, R_reduced = la.qr(A, mode = "economic")    #use SciPy's reduced QR routine to find reduced QR decomposition of A
    
    #then solve Rx = Q^Tb where R, Q are reduced R, Q for x since x is the least squares solution to Ax=b here
    #use la.solve_triangular() function to solve this because la.solve_triangular solves the equation Ax=b
    #return x once solved it:
    
    right_side = np.transpose(Q_reduced) @ b   #get right side of equation want to solve using la.solve_triangular function
    
    x_solution = la.solve_triangular(R_reduced, right_side)   #get our x solution using the given solution
    
    return x_solution
    

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    
    data = np.load("housing.npy")  #first load data from housing.npy file. Data is a matrix. First column in data matrix are x values
                                   #2nd column in data matrix are y values
                                   
    #construct the matrix A and vector b described by equation 1.2 in lab manual:
    m = len(data)              #initalize m to be the size of data (number of rows that data has)
    A = np.zeros((m, 2))       #initalize A as an array of all 0s, then will fill in later and change all values in it that need to
                               #have (m,2) inside because that is shape of A that want: want m rows and 2 columns in it
    b = np.zeros((m,1))        #initalize b as an array of all 0s, then will fill in later and change all values in it that need to
                               #have (m,1) inside because that is shape of b that want: want m rows and only 1 column in it
    #now loop through all rows in data matrix and change 1st column of A to be all x values (the 1st column of data)
    for i in range(m):
        A[i,0] = data[i, 0]    #change the values of A in its 1st column to be values in datas 1st column
        A[i,1] = 1             #change the 2nd column of A to be 1s (says: loop through each row and make the 2nd element equal 1 for each row)
        b[i] = data[i, 1]      #change the values of b to be same as values in datas 2nd column
        
    #use function from problem 1 to find the least squares solution with A, b just made:
    solution = least_squares(A, b)   #least squares returns an array: the 1st element of array is the reduced a, 2nd element of array
                                     #is the reduced b so this gives you the a, b need to make the least squares solution line
                                     #a here is the slope of line and b here is y-intercept, so 1st element of solution is slope, 2nd element is y
                                     #intercept need for plotting this line
                                    
    #plot the data points as scatter plot and plot least squares line with this scatter plot:
    #follow the green box underneath this problem in the lab manual to plot this:
    x = np.linspace(0, 16, 16)    #creates array of evenly spaced numbers in this range that will act as domain for plot
    
    plt.plot(data[:, 0], data[:, 1], "k*", label = "Data Points")             #plot data points as scatter p: our x values here are the 1st column
                                                                              #of data array and our y values to plot are the 2nd column of data
    plt.plot(x, solution[0] * x + solution[1], label = "Least Squares Fit")   #plot the least squares solution line
    #says: plot over domain specified, then plot the line where 1st element of solution is our slope of line, 1nd element of solution is y-intercept
    
    plt.legend(loc = "upper left")
    plt.xlabel("Year")
    plt.ylabel("Housing Price Index")
    plt.title("Least Squares Line with Data Points")
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    data = np.load("housing.npy")    #load the data from file 
    
    #now make the polynomials of degree 3,6,9, 12 using the Vandermode matrix stuff in lab manual
    #from equation 1.3: we want the vander matrices with the x values from data (so with 1st column of data array)
    #make vander matrices with all x values from data array: so data[:,0] gets the 1st element from each row of data (so gets
    #the whole first column of it)
    #this right here creates our A matrices (have one A matrix for each polynomial degree have, so this gives us each A need in equation 1.3)
    degree_3 = np.vander(data[:,0], 4)     
    degree_6 = np.vander(data[:,0], 7)
    degree_9 = np.vander(data[:,0], 10)
    degree_12 = np.vander(data[:,0], 13)
    
    #create our vector b need to solve equation 1.3:
    m = len(data)
    b = np.zeros(m) 
    for i in range(m):
        b[i] = data[i, 1]
        
    #now solve equation 1.3 for each A using scipy.linalg.lstsq() function, so we will get a least squares solution for each polynomial need
    #the least squares solution solves for x where the x's are the coefficients
    x_3 = la.lstsq(degree_3, b)[0]
    x_6 = la.lstsq(degree_6, b)[0]
    x_9 = la.lstsq(degree_9, b)[0]
    x_12 = la.lstsq(degree_12, b)[0]
    
    #now make polyonmials out of those coefficients just found: need polyonmials because need to graph the polynomials
    #call each polyonmial y
    y_3 = np.poly1d(x_3)
    y_6 = np.poly1d(x_6)
    y_9 = np.poly1d(x_9)
    y_12 = np.poly1d(x_12)
    
    #now do the plotting:
    plt.suptitle("Polynomials that best fit the data")
    
    x = np.linspace(0, 16, 100)    #create domain to plot all of the polynomials over: the housing data goes up to 16 on x axis
                                   #so put 16 there
    
    #plot the degree 3 polynomial:
    plt.subplot(2, 2, 1)          #create our first subplot
    plt.plot(data[:, 0], data[:, 1], "k*", label = "Data Points")  #plot the data points
    plt.plot(x, y_3(x), color = 'fuchsia', label = "Polynomial")  #plot the polynomial
    #the first says plot over our domain x, y_3(x) says to plot the polynomial over those x values in domain 
    
    plt.title("Third Degree Polynomial Least Squares Solution")
    plt.xlabel("Years after 2000")
    plt.ylabel("Housing Price Index")
    plt.legend(loc = "upper left")
       
    #plot the degree 6 polynomial:
    plt.subplot(2, 2, 2)          #create our second subplot
    plt.plot(data[:, 0], data[:, 1], "k*", label = "Data Points")  #plot the data points
    plt.plot(x, y_6(x), color = 'aqua', label = "Polynomial")  #plot the polynomial
    
    plt.title("Sixth Degree Polynomial Least Squares Solution")
    plt.xlabel("Years after 2000")
    plt.ylabel("Housing Price Index")
    plt.legend(loc = "upper left")
    
    #plot the degree 9 polynomial:
    plt.subplot(2, 2, 3)   
    plt.plot(data[:, 0], data[:, 1], "k*", label = "Polynomial")  #plot the data points
    plt.plot(x, y_9(x), color = 'lawngreen', label = "Polynomial") #plot the polynomial
    
    plt.title("Ninth Degree Polynomial Least Squares Solution")
    plt.xlabel("Years after 2000")
    plt.ylabel("Housing Price Index")
    plt.legend(loc = "upper left")
    
    #plot the degree 12 polynomial:
    plt.subplot(2, 2, 4)   
    plt.plot(data[:, 0], data[:, 1], "k*", label = "Polynomial") #plot the data points
    plt.plot(x, y_12(x), color = 'darkorange', label = "Polynomial")  #plot the polynomial
    
    plt.title("Twelfth Degree Polynomial Least Squares Solution")
    plt.xlabel("Years after 2000")
    plt.ylabel("Housing Price Index")
    plt.legend(loc = "upper left")
    
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")
    
    
# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    data = np.load("ellipse.npy")    #first load the data from the ellipse file have
    #data is array with 2 columns: the 1st column = x values , 2nd column = y values in ellipse
     
    #find parameters for ellipse: so find a,b,c,d,e parameters
    #solving something similar to equation 1.6 in lab manual, but instead we have:
    #a matrix of [x^2 x xy y y^2] that has m amount of rows in it (since from the data array, so like how first matrix in equation 1.6 goes down
    #and then have a vector with [a b c d e]^T and then those multiplied by eachother are equal to vector of only ones (with m ones in them)
    #this is because we want to solve the equation ax^2 + bx + cxy + dy + ey^2 = 1
    #AND WANT TO SOLVE FOR the [a b c d e]^T vector here: this is what the least squares solution gives you
    
    m = len(data)   #get the number of rows in data array
    b = np.ones(m)  #get vector of ones that have on right side of equation. This is our b vector
    
    #get our x and y values here
    x = data[:, 0]    #all of the x values are the first element in every row. Use splicing to get this
    y = data[:, 1]    #all of the y values are the second element in every row. Use splicing to get this
    
    A = np.column_stack((x ** 2, x, x * y, y, y ** 2))  #column stack them because the x^2 gives the whole column of x^2's from x1 to xn
                                                    # so column stack them to get what we want here. This gives me the A matrix
                                                    
    #now solve the equation Ax=b using least squares: our x here is vector of parameters that want to solve for
    parameters = la.lstsq(A, b)[0]    #they tell you to use this function to do least squares in problem 3
    
    #plot the data points:
    plt.scatter(x, y, color = 'k', marker = '.')
    #use the function given in lab manual / this file to plot the ellipse:
    plot_ellipse(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4])  #our solution from the least squares part gives us array of our a,b,c,d and e parameters in it
                                                                                             #and so access them in the array we have and plot them
    plt.title("Ellipse Least Squares Solution")   #title the graph
                
    plt.show()
    

# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    #follow psuedo code given in algorithm 1.1 in lab manual:
    
    m, n = np.shape(A)
    x = np.random.random(n)
    x = x / la.norm(x)       #by default, the norm is the 2 norm
    
    for k in range(0, N):    #this will loop through k from 0 to N-1
        x_copy = x.copy()    #make a copy of x because need this for checking the tolerance part of the problem
        x = A @ x
        x = x / la.norm(x)
        
        #this part is for the continue the loop until either the norm is less than tolerance part
        if la.norm(x - x_copy) < tol:  #want to exit for loop if the norm of x normed (what we set x to be on line 251) - original x (which is our copy of x) is less than the tolerance variable
            break                      #then exit the for loop
        
    return x.T @ (A @ x), x    #returns the eigenvalues 1st, then returns the eigenvalues 2nd
    
# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    #follow the psuedo code in algorithm 1.2 given in the problem:
    #only case when S[i] is 1x1 is the leftmost upper left entry or the mostright lower right entry = only time it is 1x1 here
    #S is upper triangular so know it's all 0s below but dont know about everything above the diagonal
    
    m, n = np.shape(A)
    S = la.hessenberg(A)
    
    for k in range(0, N):  #gets k from 0 to N-1
        Q, R = la.qr(S)
        S = R @ Q
       
    eigs = []
    i = 0
    
    while i < n:
        if i == n - 1 or abs(S[i, i + 1]) < tol:  #if S[i] is 1x1
        #the if statement: if 1x1: only happens if at very end and then need the condition for if its tol because they tell you to in the lab manual
            eigs.append(S[i][i])   #if in last spot: S[i] is the whole row, so put [-1] to get the last entry   
       
        else: #if S[i] is 2x2
            #calculate the eigenvalues of S[i].
            #we know that for a 2x2 matrix that the eigenvalues can be computed in an alternate way which is by taking the main diagonals and then
            #doing + and - (so like the quadratic formula) the squareroot of the mean squared minus the determinant of the 2x2 matrix
            #the mean of the diagonals for a 2x2 matrix is: the top left entry + bottom right entry divided by 2
            
            matrix_mean = (S[i][i] + S[i+1][i+1]) / 2  #top left + bottom right entry of the 2x2 matrix divided by 2
            matrix_det = (S[i][i] * S[i+1][i+1]) - (S[i][i+1] * S[i+1][i])  #normal determinant formula for a 2x2 matrix
            
            #find the 2 eigenvalues: for 2x2 matrix: there are always 2 eigenvalues
            eigenvalue_1 = matrix_mean + cmath.sqrt(matrix_mean ** 2 - matrix_det)   #follow the formula have described above in lines 300-303 to find the eigenvalues
            eigenvalue_2 = matrix_mean - cmath.sqrt(matrix_mean ** 2 - matrix_det)   
                      
            #append the eigenvalues of S[i] to eigs:
            eigs.append(eigenvalue_1)
            eigs.append(eigenvalue_2)
            
            i = i + 1
            
        i = i + 1
        
    return eigs
    
    
            
