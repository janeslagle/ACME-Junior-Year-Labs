# linear_transformations.py
"""Volume 1: Linear Transformations.
Jane Slagle
Math 345 Section 2
9/23/21
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time

# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    transformation_matrix = [[a, 0], [0,b]]                      #makes the stretch matrix that they give us in lab manual
    stretched_image = np.dot(transformation_matrix, A)           #stretches the matrix A given in function by transformation_matrix have
    return stretched_image

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    transformation_matrix = [[1, a], [b, 1]]                     #makes the shear matrix that they give us in lab manual
    sheared_image = np.dot(transformation_matrix, A)             #shears matrix given in function
    return sheared_image

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    transformation_matrix = [[(a * a - b * b) / (a * a + b * b), (2 * a * b) / (a * a + b * b)], [(2 * a * b) / (a * a + b * b), (b * b - a * a) / (a * a + b * b)]]
    reflected_image = np.dot(transformation_matrix, A)            #all this code reflects the matrix given in the problem
    return reflected_image

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    transformation_matrix = [[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    rotated_image = np.dot(transformation_matrix, A)             #all this code rotates the matrix given in the problem
    return rotated_image
    
# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    
    time_interval = np.linspace(0, T, 200)               #time interval [0, T]. This is what want to plot over
    arrayEarth = [[], []]                                #first array is for x values from position earth, the second array is for y values from position earth so able correctly plot
    arrayMoon = [[], []]                                 #array will need to plot the moon position on the plot make in this problem
    
    for i in time_interval:                                    #loop through all of the different times in our time interval so that can plot the earth, moon positions
        position_earth = rotate([x_e, 0], i * omega_e)         #gives you value of one x and one y (one point when plotting on graph)
        arrayEarth[0].append(position_earth[0])                #add the x values get from position_earth to x value spot of arrayEarth
        arrayEarth[1].append(position_earth[1])                #add the y values get from posiotion_earth to y value spot of arrayEartth
                   
        moonPoint = [x_m - x_e, 0]                             #want rotate vector Pm(0) - Pe(0) so first intialize the vector want to work with here
        position_moon = rotate(moonPoint, i * omega_m)         #want rotate moonPoint vector made by t omega_m radians, so do that here and set it equal to the position of moon
        moon_around_earth = position_earth + position_moon     #step 3 in problem: want translate moon vector by the earth vector, so add the two here to get the moon relative to earth
        arrayMoon[0].append(moon_around_earth[0])              #add the x values get from moon relative to earth vector to x value spot in arrayMoon so able plot correctly
        arrayMoon[1].append(moon_around_earth[1])              #do same as line 89 but with the y values now
                                                               #Do all inside for loop so that it gets it all relative to the correct time interval value
     
    plt.plot(arrayEarth[0], arrayEarth[1], 'b', linewidth = 3)           #plot the x, y values for the earth at once as a blue line
    plt.plot(arrayMoon[0], arrayMoon[1], 'orange', linewidth = 3)        #plot the x, y values for the moon at once as an orange line
    plt.axis("equal")                                                    #fix the aspect ratio of the axes
    plt.legend(["Earth", "Moon"], loc = "lower right")                   #create legend on graph plotting to label the lines corresponding to the Earth and the Moon
    plt.show()                    
    
def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    
    domain = 2 ** np.arange(1, 10)    #the domain doing multiplication and plotting over
    
    matrix_matrix_times = []          #list will put the times that need to plot in. This is list for plotting the matrix matrix multiplication times
    matrix_vector_times = []          #list will put the times that need to plot in. This is list for plotting the matrix vector multiplication times  
    
    for n in domain:
       A = random_matrix(n)           #use the random_matrix function to create A as a random matrix with n as parameter
       x = random_vector(n)           #use the random_vector function to create x as a random vector with n as parameter
       B = random_matrix(n)           #use the random_matrix function to create B as a random matrix with n as parameter
       
       start = time.time()            #use time module to figure out your start time 
       matrix_matrix_product(A, B)    #want time how long it takes to do the matrix matrix multiplication, so do the multiplication here right after you start the time
       matrix_matrix_times.append(time.time() - start)   #subtract the ending time from the start time and add the time to the list of matrix matrix multiplication times have
       
       second_start = time.time()   #do all the same stuff as in lines 144-146, but with matrix vector multiplicaion instead
       matrix_vector_product(A, x)
       matrix_vector_times.append(time.time() - second_start)
       #have lines 144-146, 148-150 inside the for loop because the matrices and vectors have n amount of things in them
       
    plt.subplot(1, 2, 1)             #plot the first subplot of the matrix vector multiplication times
    plt.plot(domain, matrix_vector_times, 'b.-', linewidth = 2, markersize = 15)
    plt.xlabel("n", fontsize = 14)
    plt.ylabel("Seconds", fontsize = 14)
    plt.title("Matrix-Vector Multiplication")
    
    plt.subplot(1, 2, 2)             #plot the second subplot of the matrix matrix multiplication times
    plt.plot(domain, matrix_matrix_times, 'r.-', linewidth = 2, markersize = 15)
    plt.xlabel("n", fontsize = 14)
    plt.title("Matrix-Matrix Multiplication")
    
    plt.show()
    
# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    
    domain = 2 ** np.arange(1, 10)  #the domain doing multiplication and plotting over
    
    product_matrix_matrix = []        #list will put the times that need to plot in. This is list for plotting the matrix matrix multiplication times with product
    product_matrix_vector = []        #list will put the times that need to plot in. This is list for plotting the matrix vector multiplication times with product
    dot_matrix_matrix = []            #list will put the times that need to plot in. This is list for plotting the matrix matrix multiplication times with dot
    dot_matrix_vector = []            #list will put the times that need to plot in. This is list for plotting the matrix vector multiplication times with dot
    
    for n in domain:
       A = random_matrix(n)         #use the random_matrix function to create A as a random matrix with n as parameter
       x = random_vector(n)         #use the random_vector function to create x as a random vector with n as parameter
       B = random_matrix(n)         #use the random_matrix function to create B as a random matrix with n as parameter
       
       start = time.time()          #use time module to figure out your start time 
       matrix_matrix_product(A, B)  #want time how long it takes to do the matrix matrix multiplication with product, so do the multiplication here right after you start the time
       product_matrix_matrix.append(time.time() - start)   #subtract the ending time from the start time and add the time to the list of matrix matrix multiplication times have
       
       second_start = time.time()   #do all the same stuff as in lines 144-146, but with matrix vector multiplicaion with product instead
       matrix_vector_product(A, x)
       product_matrix_vector.append(time.time() - second_start)
       
       third_start = time.time()    #do all the same stuff as in lines 144-146, but with matrix vector multiplication using dot instead
       np.dot(A, x)                 #do the matrix vector multiplication with dot
       dot_matrix_vector.append(time.time() - third_start)
       
       fourth_start = time.time()   #do all the same stuff as in lines 144-146, but with matrix matrix multiplication using dot instead
       np.dot(A, B)                 #do the matrix matrix multiplication with dot
       dot_matrix_matrix.append(time.time() - fourth_start)
       
    
    plt.subplot(1, 2, 1)            #plot the first subplot of all 4 execution times on a regular linear scale
    plt.plot(domain, product_matrix_vector, 'b.-', lw = 2, ms = 15, label = "Matrix-Vector with lists")
    plt.plot(domain, product_matrix_matrix, 'g.-', lw = 2, ms = 15, label = "Matrix-Matrix with lists")
    plt.plot(domain, dot_matrix_vector, 'r.-', lw = 2, ms = 15, label = "Matrix-Vector with numpy arrays")
    plt.plot(domain, dot_matrix_matrix, 'm.-', lw = 2, ms = 15, label = "Matrix-Matrix with numpy arrays")
    plt.legend(loc = "upper left")
   
    plt.subplot(1, 2, 2)            #plot the second subplot of all 4 execution times on a log scale
    plt.loglog(domain, product_matrix_vector, 'b.-')
    plt.loglog(domain, product_matrix_matrix, 'g.-')
    plt.loglog(domain, dot_matrix_vector, 'r.-')
    plt.loglog(domain, dot_matrix_matrix, 'm.-')
   
    plt.show()
        
