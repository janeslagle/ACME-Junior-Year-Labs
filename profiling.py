# profiling.py
"""Python Essentials: Profiling.
Jane Slagle
Math 347 Section 1
1/4/21
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
from numba import jit
import time
from matplotlib import pyplot as plt

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    #first read from the file so that can get the actual triangle:
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]           #data is the triangle
                        for line in reversed(infile.readlines())]  #reverse the triangle so that data[0] is the bottom
                                                                   #of triangle
    
    #need to step through triangle from bottom to top:
    for i in range(len(data)):   #loop through triangle: first loop through rows
        for j in range(len(data[i]) - 1):  #then loop through each column (which is each entry so data[i] to loop through
                                           #each row. Need -1 because starting at zero and want to go to the top, so need
                                           #a -1 there
            num_1 = data[i][j]
            num_2 = data[i][j + 1] 
            max_num = max([num_1, num_2])   #get the max of each 2 number pair in the row of triangle. max can only be
                                            #used on a list
            
            #data[i + 1][j] gets the parent of the max of the 2 nums by getting the parent in row above
            
            data[i + 1][j] += max_num  #replace the parent num with the added number want
            
    return data[-1][0]  #want to return the top entry (which is the max sum) and reversed it so top entry is data[-1]
                        #and to return the actual entry we have to put data[-1][0]            


# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]   #put 2 in there since know 2 is prime # so don't need to check if it is
    current = 3  #start at 3
    
    length = 1         #will make it a lot faster if just have this
    while length < N:
        isprime = True   
        square_root = current ** 0.5   #need for one of the conditions they tell us to check
        for i in primes_list:  #loop through list of primes 
            if i > square_root:  #one of the conditions need to check
                break            #want to break if greater than the sqrt because know its not prime
            if current % i == 0: 
                isprime = False  
                break           #just immeditately break bc know its not prime already if goes through this if statement
                                #so will save a lot of time to do this
        if isprime:             
            primes_list.append(current)
            length += 1         
        current += 2   #avoids all even numbers so that don't have to check as many numbers
        
    return primes_list
           
    
# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    #use argmin because want index of column of A that is closest to x
    #take the transpose of A and then subtract x
    
    return np.argmin(np.linalg.norm(A.T - x, axis = 1))  #need axis = 1 bc is columns since A is a column vector
   
   
# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
        
    #use enumerate: gives you the index and the name of the thing at the index
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"   #will use this alphabet in the dictionary
    dictionary = {letter: index + 1 for index, letter in enumerate(alphabet)}  #format of dict - have key:value and then have to say
                                                                               #where the index will come from and enumerate the 
                                                                               #alphabet bc want to associate each letter with a number
                                                                               #and have +1 bc dont want index to start at 0
    #have to add each letter's index times its namescore
    total = 0
    for ind, name in enumerate(names):   #to get namescore: need to add up all the letters together
    #need to look up specific letters
        name_value = 0                                  #want name_value to start over for each letter so that can get the specific
                                                        #name value for each letter
        name_value = sum([dictionary[j] for j in name]) #for getting the letters
        total += name_value * (ind + 1)  #need to do ind+1 for same reason did ind+1 up above
                                         #want to return the total: so need to add them all together
    return total
                                                                             

# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    F_1 = 1    #need to initialize the 1st two fibonacci numbers
    F_2 = 1
    
    while True:          #have in while True loop so that can keep getting fibonacci numbers forever
        yield F_1        #need yield to return F_1 bc with yield: return 1 value at a time, so need to return F_1
                         #before change it 
        F_1 = F_1 + F_2  #with yield: starts right here again: gets the next number
         
        yield F_2        #F_2 is the next number so yield it
        F_2 = F_1 + F_2  #get the next number. And then will go back to top of while loop and return this next fib num
                         #just found and then will get next number, yield (return) it, etc.
    
#test it by doing: fib = fibonacci() in terminal
#and then doing next(fib)


def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    #first have to get the numbers from the generator: need to start it off outside the loop so that have a condition
    #to check
    index = 1             #initialize the index: our 1st fibonacci number is F_1 so our index starts at 1, not 0
    fib = fibonacci()     
    fib_num = next(fib)   #get the next fibonacci numbers from generator
    
    while len(str(fib_num)) < N:   #need str(fib_num) bc cant take len of number: so convert it to string
        index += 1                 #increment the index each time go through while loop bc get another number each time
                                   #go through the while loop
        fib_num = next(fib)   #get the next number again
   
    return index
       

# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    #following the steps given in lab manual:
    #1) start with all of integers from 2 to N
    nums = np.arange(2, N)  #store the ints in a list
    
    while len(nums) > 0:
        yield nums[0]    #yield 1st entry in list
        
        #find the numbers in our list nums that are relatively prime to 1st entry in list (so that aren't divisible by
        #the 1st entry). Use a mask to do this
        mask = nums % nums[0] != 0     #check if divisible by 1st entry. mask is a boolean so when do nums[mask] it only
                                       #keeps the indices from nums list where the mask condition is true
        
        #set our list to only be the entries in list that weren't divisible by 1st entry
        nums = nums[mask]
    
        #the while loop is how it repeats like how they say to in the steps in the lab manual
        
# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product
    

#use Numba by putting @jit right before the function definition. Just copy code from matrix_power and put jit in front
#because want the Numba-enchanced version of matrix_power and putting jit in front of it is how you do that
@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product
    

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    #first test numba on a small matrix first so that numba can compile and so that numba doesn't have to take the time to compile
    #when I acutally want to time it below. So need to do this test first so that it graphs it correctly
    small_test = np.random.random((2,2))
    test_output = matrix_power_numba(small_test, 1)
    
    power_times = []   #list to store the times for matrix_power
    numba_times = []   #list to store the times for matrix_power_numba
    linalg_times = []  #list to store np.linalg.matrix_power() times
    
    #want to do everything for m = 2^2,2^3,...,2^7 so need to do everything in a for loop, etc.
    #first get the m values want and put them in a list so that can loop through them
    m_vals = [2**i for i in range(2,8)]
    
    for m in m_vals:
        A = np.random.random((m,m))    #generate random mxm matrix A with np.random.random()
        
        #time matrix_power(A, n):
        start_1 = time.time()
        output_1 = matrix_power(A, n)
        end_1 = time.time() - start_1
        power_times.append(end_1)
        
        #time matrix_power_numba(A, n):
        start_2 = time.time()
        output_2 = matrix_power_numba(A, n)
        end_2 = time.time() - start_2
        numba_times.append(end_2)
        
        #time np.linalg.matrix_power(A, n):
        start_3 = time.time()
        output_3 = np.linalg.matrix_power(A, n)
        end_3 = time.time() - start_3
        linalg_times.append(end_3)
        
    
    #plot the times on log-log plot
    plt.loglog(m_vals, power_times, base = 2, color = "violet", label = "Python: matrix_power()")
    plt.loglog(m_vals, numba_times, base = 2, color = "indigo", label = "Numba: matrix_power_numba()")
    plt.loglog(m_vals, linalg_times, base = 2, color = "orangered", label = "NumPy: np.linalg.matrix_power()")
    plt.legend(loc = "upper left")
    plt.xlabel("Matrix size")
    plt.ylabel("Time (seconds)")
    plt.title("Speed of Matrix Multiplication Functions")
    plt.show()
 

    
