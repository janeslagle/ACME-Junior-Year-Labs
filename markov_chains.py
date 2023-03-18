# markov_chains.py
"""Volume 2: Markov Chains.
Jane Slagle
Math 321 Section 2
11/4/21
"""

import numpy as np
from scipy import linalg as la
import random


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        matrix A ((n,n) ndarray) - column stochastic matrix given as a parameter in the problem
        list of labels (list(str)) - list of the states given as a parameter in the problem
        dictionary (dict) - maps the state labels to the row/column index that they correspond to in A
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        #allclose compares each entry in a matrix or array and if all entries are close enough to a tolerance within 
        #eachother then it says they're equal and returns true
        if not np.allclose(A.sum(axis=0), np.ones(A.shape[1])):     #check if A is column stochastic. This condition was given in reading quiz for lab
            raise ValueError("The matrix is not column stochastic")
            
        m,n = np.shape(A)           #get the number of columns and number of rows in A
        if m != n:                  #check if A is square
            raise ValueError("The matrix is not square")
            
        #save A, a list of labels and a dictionary as attributes 
        self.A = A
        self.dict_labels = dict()   #this maps the state labels to the row/column index that they correspond to in A.
                                    #the key is the label and the value assigned to each key is its index. Each
                                    #key only has 1 value assigned to it each
                                    
        #create the states attribute:
        if states != None:     
            self.states = states    #just save the states attribute as what given as states parameter if weren't given None
        else:                       #BUT if no state labels given, use the labels [0 1 ... n-1]
            lst = [k for k in range(A.shape[0])]  #have label for each row of matrix A, so loop through and add label for each
            self.states = lst                     #of A's rows (so the shape of A). Have shape[0] because just want the # of rows have
                                                  #since have range of shape of A: the range for i will go from 0 to n-1 like want it to
                 
        #construct the dictionary using list comprehension
        states = self.states        #need this variable here because if just had self.states everywhere currently have a states
                                    #in the dictionary construction below, it will alter self.states for some reason and mess 
                                    #everything up 
        self.dict_labels = {states: i for i, states in enumerate(states)}  #everything before the colon is the key (the label)
                                                                           #everything after is for getting index i
        #construct the dictionary like this because its of the form: {label, index}
        #enumerate(states) creates and assigns an index for all states so i in enumerate(...) will make i the index since states is a list
        #so enumerate assigns an index to each label
        #for loop part loops through all labels, getting an index for each one
        #enumerate returns index, value: so want to have i, states above on line 71
                                                                                        
    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        #use dictionary to determine which column of A the given state corresponds to:
        column = self.dict_labels[state]  #state is key in dictionary: each key has only 1 value, so calling dictionary
                                          #at this state key value will get the index that corresponds to state (which is the column)
        
        #draw from corresponding categorical distribution: use np.random.multinomial function like says in lab manual:
        #have 1 because want make cateorgical draw (told in lab manual)
        draw = np.random.multinomial(1, self.A.T[column])   #want the column here so do A.T[column] because that gives us the actual column
                                                            #if just did A[column]: it would give us the row at that index so just take transpose
                                                            #to fix this issue
        #draw returns np.array with 1 where winner is, 0 where winner is not (winner is which one is chosen)
        #want index of that chosen one (of whichever one was 1) so use np.argmax
        new_state_index = np.argmax(draw)                   #argmax returns index of biggest thing in array which will be our 1 here so using
                                                            #argmax will give index of which state want to transition to
        #want return value at this index: so loop through dict and see which key is defined with that index
        for key in self.dict_labels: 
            if self.dict_labels[key] == new_state_index:   #check if each key value in the dict has this index
                new_state = key
       
        return new_state
        

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        results = [start]  #list we want to return, initialize it with start so that we start at this specified start state
        
        for i in range(N - 1):  #have N-1 and not N because the start one counts as the 1st one (because we have start in our list already)
            state_label = self.transition(start)   #we want to start our transition at the start label
            results.append(state_label)
            
        return results        
        

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        results = [start]
        
        while start != stop:   #want to transition from state to state until get to the specified stop state
            state_label = self.transition(start)
            results.append(state_label)
            start = state_label           #need to update what our start index is for when re-enter the while loop
            
        return results
        
      
    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        #make random state distribution vector x
        x_0 = np.random.rand(self.A.shape[0])  #A.shape returns 2 numbers, but A square, so just return one of those numbers
                                               #want x be same shape as A because want to do Ax
                                               #by default: np.random.rand is positive always, so know entries of x are nonnegative
        #x_k needs to sum to 1 so divide each entry in x_0 by the sum of all entries in x_0
        x_k = x_0 / sum(x_0)
                                             
        k = 0      #want to keep track of k (number of times go through while loop because have to check if k exceeds maxiter later)   
        #want to do this until ||xk-1 - xk|| < tol:
        #told in lab manual that A times x_k gives you x_k+1 (the next vector)
        while la.norm(x_k - (self.A @ x_k), ord = 1) >= tol:    #need ord=1 because want 1 norm and default norm is 2 norm
            x_k = self.A @ x_k                                  #need update x_k for our next time we go through the for loop
            k = k + 1
            #so process of while loop is: does norm once, then checks the condition of if k > maxiter
            if k >= maxiter: 
                raise ValueError("A^k does not converge, k exceeds the max iterations")
                
        #return approximate steady state distribution x of A. This is our x_k vector (the normed nonegative vector)      
        return x_k
          

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        matrix: transition matrix
        dictionary: same idea as the MarkovChain dictionary made
        states: has the labels (the words from our file we read) in it
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        #follow Algorithm 1.1 in lab manual:
        
        #read the file
        with open(filename, 'r') as info:
            s = info.read()                #s has contents of file
            
            #split s into list of words:
            words = s.split()              #the words are already in a list: each word is an element
            
            #split s into list of sentences:
            sentences = s.split('\n')      #the sentences are already in a list: each sentence is an element
        
        #make set of unique words    
        unique_list = []     #initialize list that will hold all the unique words from the file in it
        for word in words:   #loop through all the words have from file
            if word not in unique_list:   #only add the word if it is not already in the list. This will create list of unique words
                unique_list.append(word)
        
        #add labels "$tart", "$top" to set of state labels:
        unique_list.insert(0, "$tart")    #unique list has all words in it and words are our labels, so add these other labels
                                          #to the unique list since it has all the labels in it
                                          #want start one to be 1st thing in list, so use insert with 0 to do that
        unique_list.append("$top")        #want stop to be the last thing in list, so just append it to our list
        
        #initialize square array of zeros to be transition matrix
        #the matrix needs to be the same size as our unique_list
        m = len(unique_list)        
        matrix = np.zeros((m,m))   #transition matrix: have column, word for each word (words are labels), so want it to be mxm
        
        #want to loop through all sentences, so loop through sentences list:
        for sentence in sentences:
            sent_words = sentence.split()       #split sentence into list of words
            sent_words.insert(0, "$tart")      
            sent_words.append("$top")
            
            for i in range(len(sent_words) - 1):    #want loop through new list of words made
                #have -1 because dont want to go all way to end since getting the next one
                #go to each word: go to the next one, and then find where the 2 words line up in the matrix and put a 1 in that spot
                #so have a 1 between whichever words are next to each other
                
                wordIndexX = unique_list.index(sent_words[i])       #gets index of where this word is in our unique_list
                wordIndexY = unique_list.index(sent_words[i + 1])   #our index is word here. Gets the word in the next column over
                
                matrix[wordIndexY, wordIndexX] += 1                 #goes to that spot in our transition matrix and puts a 1 there 
                
                if i == len(sent_words) - 2:        #len(sent_words) - 2 is index of 2nd to last column
                    matrix[len(unique_list) - 1][wordIndexY] += 1   #len(unique_list) - 1 gives you last row
                                                                    #wordIndexY gives you the thing in the last column because wordIndexY
                                                                    #will be the column after 2nd to last column: so last column
                   
        column_sums = [sum([row[i] for row in matrix]) for i in range(0, len(matrix[0]))]   #gets column sums
        matrix = matrix / column_sums        #normalize each column by dividing by the column sums
        
        #save same attributes as constructor made in MarkovChain class
        self.A = matrix
        self.states = unique_list
        self.dict_labels = {}
        
        labels = self.states 
        self.dict_labels = {labels: i for i, labels in enumerate(labels)} 
        

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        #use problem 3 function:
        sentence = self.path("$tart", "$top")
        
        sentence.remove("$tart")
        sentence.remove("$top")
        
        string_sentence = ' '.join(sentence)  #converts the list to a single, space-separated string
        
        return string_sentence
        
        
        
