# dynamic_programming.py
"""Volume 2: Dynamic Programming.
Jane Slagle
Volume 2 lab
4/5/22
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    V = dict()                                     #dictionary to store V in since in equation 10.1 need to call the previous V in it 
                                                   #V is the expected values of choosing candidate t where t is the key
    V[N] = 0                                       #the value of V at N is always 0, so just define the Nth key of dict to have value 0
    
    #t starts at N and iterates down to 1
    for t in range(N-1, 0, -1):    #only loop through from N-1 down to 1 bc already defined what the Nth value is
        V[t] = max((t)*V[t+1]/(t+1) + (1/N), V[t+1])     #equation 10.1
        if V[t] == V[t+1]:          #if have case where start repeating, want return the 1st one where start repeating
            return V[t+1], t+1
    
    return V[1], 1                  #otherwise: will ALWAYS return t=1 one (that will always be the max)
        
# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    opt_per = []   #list to store the optimal stopping % of candidates (t_0/N)
    max_prob = []  #list to store the maximum probabilities 
    
    #run function from prob 1 for N = 3,4,...,M   
    for N in range(3, M+1):
        opt_per.append(calc_stopping(N)[1] / N)   #want to plot t_0/N values
        max_prob.append(calc_stopping(N)[0])      #want to plot the expected values V(t_0)
        
    #plot everything now:
    plt.plot(np.arange(3, M+1), opt_per, color = "navy", label = "optimal stopping %")
    plt.plot(np.arange(3, M+1), max_prob, color = "deeppink", label = "maximum probability")
    plt.legend(loc = "best")
    plt.title("Optimal stopping % and max prob against N")
    plt.show()
    
    return calc_stopping(M)[1]/M    #return optimal stopping % for M. remember optimal stopping % is t_0/N
   
# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    W = 1
    C = np.zeros((N+1, N+1))     #initialize consumption matrix C as all 0s
    w = np.linspace(0, 1, N+1)   #make vector w given in lab manual. this is the pattern it follows
    
    #now make the C matrix:
    for i in range(N+1):         
       for j in range(i):      
           C[i][j] = u(w[i] - w[j])   #this is how the consumption matrix is defined
    
    return C
   
# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    #prob 4 stuff:
    A = np.zeros((N+1, T+1))     #A is an N+1 x T+1 
    w = np.linspace(0, 1, N+1)   #w is the same as in prob 3
    P = np.zeros((N+1, T+1))   #in prob 4: want return matrix of 0s for policy matrix P
    CV = np.zeros((N+1, N+1))    #initialize CV matrix as all 0s. make CV matrix in prob 5 
    
    #want make matrix A. A has 0s everywhere except last column
    for i in range(N+1):
        A[i][-1] = u(w[i])       #want change only the last column of A
        P[i][-1] = w[i]          #prob 6 stuff: get last column of P. last column of P will be w
       
    #prob 5 stuff: get everything for A, P except for the last column's
    for t in reversed(range(T)):   #need to make the CV matrix. do reversed range because start at the end and go down to 1. want to skip the last column
        for i in range(N+1):
            for j in reversed(range(i+1)):    
               CV[i][j] = u(w[i] - w[j]) + B*A[j][t+1]    #equation 10.7
               
            #told in lab manual that P_it = w_i - w_j so only need to loop through i, t to get P
            P[i][t] = w[i] - w[CV[i].argmax()]   #told that j is min(j) where CV[i][j] >= CV[i][k]. this will be given by CV[i].argmax()
            A[i][t] = max(CV[i])   #equation 10.7
            
    return A, P

# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    #step 1: get A, P matrices from eat_cake function
    A, P = eat_cake(T, N, B, u)
    
    #step 2: initialize C as empty list
    C = []
    
    #step 3: n is number of rows
    n = N
    
    #step 4: loop through and make C matrix. t represents number of columns
    for t in range(T+1):
        C.append(P[n, t])
        n -= int(C[-1]*N + 0.5)
        
    return C   #C should add up to 1
    
def testing():
    #print(eat_cake(3, 4,.9))
    
    print(find_policy(3, 4, .9))
    print(find_policy(5, 6, .6))
    
    
