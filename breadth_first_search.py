# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Jane Slagle
Math 321 Section 2
10/27/21
"""

import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import networkx as nx

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in self.d:      #check if n is in graph or not (so check if d is in dictionary d since d has all of the nodes 
                                 #in the graph in it
            self.d[n] = set()    #create node n in the dictionary, but dont want the node to have any edges so set it equal to empty set
                                 #keys in the dictionary are the edges

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        #if nodes u and v are not already present, use add_node function just wrote to add u and v to the graph
        self.add_node(u)
        self.add_node(v)
        
        #add an edge between node u and node v:
        self.d[u].add(v)   #in our dictionary at node u: add v to create edge between u and v
        self.d[v].add(u)   #in our dictionary at node v: add u to create edge between v and u
   
    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        #only enter if statement if node n exists in dictionary (so if n is in the graph)
        if n in self.d:  
            self.d.pop(n)      #remove node n from the graph
            for i in self.d:   #loop through all of the edges (keys) in the graph
                self.d[i].discard(n)   #and then remove each node that is connected to n (so all of the edges from n). In lab manual: use discard
                                       #not remove because it removes element from set without raising key error
        else:  
            raise KeyError("the node n is not in the graph")   #only enter this else statement if the node is not in the graph and raise the 
                                                               #KeyError like want
            

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        #check if u in graph or not. Raise key error if you they aren't
        if u not in self.d:
            raise KeyError("The node is not in the graph")
         
        #now check to see if the connection to v exists:
        elif v not in self.d[u]:   #if u, v are connected then v will be in the key of u so this will check if the connection exists 
            raise KeyError("This connection does not exist so cannot remove it")
            
        #will enter this else statement only if edge is between the 2 nodes, so can remove it all now
        else:
            self.d[u].remove(v)    #remove the node v from the key of u (so remove the edge connecting u and v, so this removes the connection)
        
        #now do all of the same stuff but for node v:    
        if v not in self.d:
            raise KeyError("The node is not in the graph")
        elif u not in self.d[v]:   
            raise KeyError("This connection does not exist so cannot remove it")
        else:
            self.d[v].remove(u)   
           

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        #check if source node is not in the graph. Raise KeyError if it isn't
        if source not in self.d:
            raise KeyError("The source node is not in the graph")
            
        #traverse the graph with a BFS until all nodes have been visited:
        V = []          #told in lab manual need to initalize all of this for the BFS stuff
        Q = deque()
        M = set()
        
        #to begin search: add source node to Q and M
        Q.append(source)
        M.add(source)      #add stuff to set by using .add
        
        #NOW: do the next steps until Q is empty, so have a while loop that has condition that checks if length of Q is 0 (which will be true
        #when Q is empty)
        while len(Q) != 0:
            current = Q.popleft()   #pop node off of Q
            V.append(current)   #append current to V
            
            #add the neighbors of current node that are not in M to Q and M
            for node in self.d[current]:  #loop through all of edges of the node current
                if node not in M:         #want the neighbors of current node that are not in M (node are the neighbors of current here)
                    Q.append(node)
                    M.add(node)
                    
        return V    #return V since it is the list of nodes visited
        
        #when test it: the source node is the one start at and then in the list V returned: it needs to have all of the other nodes in the graph
        #in it (since all nodes must be visited) and the first noed in it must be the source one (since that is one start at always with
        #this function
        

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        #because of the nature of BFS, the first path it finds is the shortest path
        
        if source not in self.d:
            raise KeyError("The source node is not in the graph")
        elif target not in self.d:
            raise KeyError("The target node is not in the graph")
            
        visited = {}      #need a dict because need a key-value pair mapping here. Key-value pair means that only have node - node, node - node, etc.
                          #so this means that each node (key) in the dictionary is only connected to one node
        path = [target]   #list want to return. Lab manual says to start at target and then loop through dict until get to source code
                          #so initialize this list with target in it since want to be starting at target here
             
        #have same process used in problem 2, but need to modify it to match this problem   
        V = []  
        Q = deque()
        M = set()
       
        Q.append(source)
        M.add(source)     
        current = source  #make current equal the source node so that start at the source node and then keep going in while loop until 
                          #reach our target node
        
        while current != target: 
            current = Q.pop()  
            V.append(current)   
            
            for node in self.d[current]: 
                if node not in M:     
                    Q.append(node)
                    visited[node] = current  #says: put the key-value pair mapping into the dict. pred[node] will create dict entry for node
                                             #and where it came from (so like the path)
                    M.add(node)
       
        #this next part is creating the shortest path BUT going from target to source (so reverse of what we want to return)
        #so actually adding our nodes to the list returning here
        temp = target            #store target value in a temp variable (because want to start at target here)
        while temp != source:    #add nodes to our list until we reach the source node (then will have nothing else to add)
            temp = visited[temp]  #get where visited so far using our dict
            path.append(temp)     #add the nodes to our list: adding them in order of target to source node
            
        return path[::-1]   #return it reversed because found shortest list from target to source but want return from source to target
       
       
# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        
        #want to save all of these as attributes:
        self.titles = set()   #initialize a set for movie titles
        self.actors = set()   #initialize a set for actor names
        self.G = nx.Graph()   #initialize an empty NetworkX graph
        
        #read the file:
        with open(filename, 'r') as info:
            movies = info.readlines()    #read file line by line because each line is movie followed by list of its actors
            
            for movie in movies:                     #loop through all the movies
                data = movie.strip().split('/')      #data is list contains split up of all the movies and all the actors
                                                     #Split by backslahes because told in lab manual that's how thye split it in the file
                                                     
                #first element in data list is movie title, all elements after the 1st one are the actors in that movie
                self.titles.add(data[0])             #so add the movie title (1st element of data) to our set of movie titles made above
                
                #now add the actors in each movie to set of actors
                for actor in data[1:]:   #loop through all actors. Start at 2nd element, go through the rest in data list
                                         #because that will give you all of the actor names
                    self.actors.add(actor)   #add the actor name to our list of actor names made above
                    self.G.add_edge(actor, data[0])  #add edge btw movie and each actor to our graph (so add to graph attribute made above)
                    
                    
    # Problem 5e
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        #use nx.shortest_path function to find the shortest path btw source, target nodes (so btw the actors)
        #this shortest path function returns the actor, movie name in that order
        
        #so 1st find this shortest path:
        shortest_path = nx.shortest_path(self.G, source, target)  #says to get the shortest path btw source, target where source, target
                                                                      #come from our graph self.graph
        #now find the length of this path (the num of steps btw them). Use another nx function to do this:
        shortest_path_length = nx.shortest_path_length(self.G, source, target)  
        
        #BUT: shortest path length gives the length of shortest path BUT does it for ALL of the movies AND the actors, BUT only want
        #actors here and remember that shortest_path returns actor, movie, actor, movie, etc. so we know that the num of actors
        #will be half of what is returned SO want to return the length divided by 2 so that only get the length of actors
        
        return shortest_path, (shortest_path_length) // 2
     
     
    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        #want to find shortest path lengths of every actor in collection to specified actor
        #find all the path lengths at once using nx function
        paths = nx.shortest_path_length(G = self.G, target = target)  #will find all lengths for every actor if don't give source parameter
        
        list_lengths = []   #will put our actor path lengths into this list
        
        for actor in self.actors:   #loop through all actors have
            list_lengths.append(paths[actor] // 2)    #gets the path length of every actor and puts it into our list of lengths. MAKE SURE
                                                      #divide it by 2 to get rid of the movie lengths. Dividing by 2 will give you only the
                                                      #actor lengths
          
        #now plot the distribution of path lengths (plot a histogram graph)
        plt.title("Distribution of Path Lengths")
        plt.hist(list_lengths, bins = [i-.5 for i in range(8)])  #have list_lengths there because want to plot the path lengths 
        plt.xlabel("Path Length")
        plt.ylabel("Number of Actors")
        plt.show()
        
        #want to return the avg path length: so sum all of the lengths together and then divide by the total length of all the paths
        #don't have to divide by 2 here bc already divided by 2 in line 304 when found the length of actors 
        average_path_length = sum(list_lengths) / len(list_lengths)
        
        return average_path_length
        
        
