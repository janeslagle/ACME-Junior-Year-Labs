# solutions.py
"""Volume 1: The Page Rank Algorithm.
Jane Slagle
Volume 1 lab
2/28/22
"""

import numpy as np
from scipy import linalg as sc
from scipy.sparse import linalg as la
import networkx as nx
from itertools import combinations

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        #modify A so that no sinks: by making all entires in b column 1s
        A[:, A.sum(axis=0) == 0] = 1   #loop through all rows, change any columns that sum to 0 to be filled with 1s
       
        #find A_hat:
        A_hat = A / np.sum(A, axis = 0)  #divide matrix A by the sum of each column: because want to normalize A
        self.A_hat = A_hat               #save as attribute
        
        #use [0, 1,...,n-1] as labels if none:
        if labels is None:
            self.labels = [str(i) for i in range(0, len(A))]
        else:
            #raise ValueError if number of labels != number of nodes in graph (there are n nodes in graph: the a,b,c,d)
            if len(labels) != len(A):
                raise ValueError("num of labels not equal to num of nodes in graph :/")
            self.labels = labels
            
    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        A_hat = self.A_hat
        n = len(A_hat)
        
        #solve for p:
        I = np.identity(n)
        left = I - (epsilon*A_hat)
        right = ((1 - epsilon)*np.ones(n))/n
        
        #need solve for p using la.solve since have to divide by matrix
        p = sc.solve(left, right)   
        
        #now make dictionary:
        dictio = dict(zip(self.labels, p))   #zip combines the labels, p and then make it into a dictionary
         
        return dictio
        
    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        A_hat = self.A_hat
        n = len(A_hat)
        E = np.ones((n,n))    #E is nxn matrix of 1s
        
        B = epsilon*self.A_hat + ((1-epsilon)/n)*E    #make B
        
        p = la.eigs(B,1)[1].real
        
        p /= p.sum()            #normalize with 1 norm bc vector is in R
        
        dictio = dict(zip(self.labels, p)) 
         
        return dictio
        
    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        n = len(self.A_hat)
        p0 = []
        for i in range(n):    #make p0 
            p0.append(1/n)
        
        for i in range(maxiter):                          #iterate until reach maxiter
            p1 = (epsilon*self.A_hat)@p0 + ((1-epsilon)/n)*np.ones(n) 
            if np.linalg.norm(p1 - p0, ord = 1) < tol:    #break if norm is less than tol
                break
            p0 = p1                                       #update p0 for next time go through for loop

        dictio = dict(zip(self.labels, p1))   
        return dictio

# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    #read everything from file:
    with open(filename) as thefile:
        data = thefile.readlines()    #data is everything from file
   
    #get all of the individual lines from file: strip all whitespaces, separate by slashes
    lines = [line.strip().split(",") for line in data[1:]]  #want skip the first line
    
    teams = set()                    #initialize set of teams as empty set
    for games in lines:              #now loop through all of the indiv lines: each line is a game between 2teams
        for game in games:           #loop through each game
            teams.add(game)           #add each of the indiv teams to the set of teams
    uniq_teams = sorted(teams)      #now sort the teams. since they are a set, teams will be unique since sets dont have any duplicates

    n = len(uniq_teams)
    A = np.zeros((n, n))              #initialize matrix A as all zeros. want A to be nxn matrix
    
    #make adjacency matrix A:
    index = {uniq_teams[j]: j for j in range(n)} #each team is a node, and want indices of matrix A's columns, rows to correspond to nodes
    for game in lines:                #loop through all of the games
        A[index[game[0]], index[game[1]]] += 1   #want order as losers to winners, 2nd team is losers, first team is winners
                                                 #will have a 1 if the 1st team beat the 2nd, and way each line is formatted, the 2 teams
                                                 #played each other, so just loop through each and make the corresponding links btw the 2 teams   
    #get the PageRank values:
    class_obj = DiGraph(A, uniq_teams)    #make the class object
    pagerank_vals = class_obj.itersolve(epsilon)      #need pass epsilon into itersolve
    actual_ranks = get_ranks(pagerank_vals)
    
    return actual_ranks    #return ranked list of webpage IDs

# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    #initialize a DiGraph object:
    graph = nx.DiGraph()
    
    with open(filename, encoding="utf-8") as readfile:     #first have to read the file
       for line in readfile:      #now loop thorugh the lines in file: each line corresponds to a movie
           #only thing want from file is the actors, so get all the actors from each line
           actors = line.strip().split('/')[1:]  #to only get the actor names: strip all whitespace from the line, split at the slashes
                                                 #since each actor name is separated by slash, and start after the movie title
                                                 #for each line: actors is redefined as list of actors for the specific movie on that line
           
           #want weights between 2 of actors: look at each combo of 2 actors within movie at line we're on                                     
           for act1, act2 in combinations(actors, 2):    #only 2 actors can be connected at a time
               #want actor 2 to be listed first, so have order be act2, act1 when add edges and weights:
               
               #first check if the 2 actors we're looking at are already connected:
               if graph.has_edge(act2, act1) == False:   #connect them if they aren't already connected
                   graph.add_edge(act2, act1, weight = 1)   #add an edge between the 2 actors: initialize weight as 1 since just made it so 
                                                            #there is one edge between them
               else:                                     
                   graph[act2][act1]["weight"] += 1      #if already edge between 2 actors: take weight and add 1 bc weight is number of 
                                                         #connections between these 2 actors
    
    #compute PageRank values of the actors, so of the graph: this gives a dictionary which need as parameter for prob 3
    rank_acts = nx.pagerank(graph, epsilon)
    
    return get_ranks(rank_acts)
                 
#function to test everything:
def testing():
    D = DiGraph(np.array([[0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 0]]))   #create class object with matrix A
    
    #test prob 1:
    """print(D.A_hat)
    print(D.labels)"""
    
    #test prob 2:
    #print(D.linsolve())
    #print(D.eigensolve())
    #print(D.itersolve())
    
    B = np.array([[0.,1.,1.,0.], [1.,0.,1.,0.], [0.,0.,1.,1.], [0.,0.,1.,1.]])
    print(DiGraph(B).eigensolve(epsilon=.62))
    print(DiGraph(B).linsolve(epsilon=.62))
    print(DiGraph(B).itersolve(epsilon=.62))
    
    print("OTHER TEST")
    A = np.array([[1, 3, 1], [1, 3, 4], [1, 3, 4]])
    print(DiGraph(A).eigensolve(epsilon=.8))
    print(DiGraph(A).linsolve(epsilon=.8))
    print(DiGraph(A).itersolve(epsilon=.8))
    
      
