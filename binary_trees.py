# binary_trees.py
"""Volume 2: Binary Trees.
Jane Slagle
Math 321 Section 2
10/7/21
"""

#These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import time
import random
import numpy as np
from matplotlib import pyplot as plt

class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        
        def recursive_step(node):        #define a function within this method that checks single node for the data
            if node is None:             #first check if the list is empty
                raise ValueError("The list is empty or the node is not in the list")
            elif node.value == data:     #then check if the node contains the data
                return node              #if the node contains the data then return the node
            else:                        #if the node doesn't contain the data, then call this function recursively again on the next node
                return recursive_step(node.next)    #calling recursively on next node will go through all the nodes and see if they contain data
            
        return recursive_step(self.head) #told want to call this inner step function on the head node


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
            
        """
        n = BSTNode(data)   #create new BSTNode containing the data
        
        def find_and_link(current):                 #current is the current node we are at in the tree
            if data == current.value:               #raise value error if the node containing data already in tree
                raise ValueError("The node is already in the tree, no duplicates!")
            if data < current.value:                #check if the new node will be on the left of parent
                if current.left == None:        #only add the new node to the left if there is nothing in the left child spot
                    current.left = n 
                    n.prev = current            #link it back to its parent
                else:
                    return find_and_link(current.left)   #if left spot is not empty then run through again to check if data is less than it
            else:              
                if current.right == None:
                    current.right = n
                    n.prev = current
                else:
                    return find_and_link(current.right)
                                           
        
        if self.root is None:   #check if tree is empty
            self.root = n       #assign the root attribute to this new node
        else:
            return find_and_link(self.root)   #start the find and linking stuff at the root of the tree
            
 
    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
            """
            
        if self.root is None:   #check if the tree is empty
            raise ValueError("The tree is empty")  
        current = self.find(data)            
        if current.left == None and current.right == None:   #case where data is leaf node
            if current.prev == None:                         #case where the target (which is current here) is the root
                self.root = None                             #remove the root by setting it's value equal to none
            elif current.prev.left == current:                 #if not root, then has a previous: checks if target is to left of parent
                current.prev.left = None
            elif current.prev.right == current:                #if target is to right of parent
                current.prev.right = None
                    
        elif (current.left != None and current.right == None):   #if target has one child to the left
            if current.prev == None:                          #case where the target is the root
                self.root = current.left                      #removing the root so have make the left child be the root now
                self.root.prev = None                         #actually remove the target now
            elif current.prev.left == current:                #case where target is left child of its parent
                current.prev.left = current.left
                current.left.prev = current.prev
            elif current.prev.right == current:
                current.prev.right = current.left
                current.left.prev = current.prev
                        
        elif (current.left == None and current.right != None):   #if target has one child to the right
            if current.prev == None:
                self.root = current.right
                self.root.prev = None
            elif current.prev.left == current:                #case where target is left child of its parent
                current.prev.left = current.right
                current.right.prev = current.prev
            elif current.prev.right == current:
                current.prev.right = current.right
                current.right.prev = current.prev
       
        else:   #if target has two children              
            #want to remove current here
            target = current.left
            while target.right is not None:    #find the most right node that have to swap with the root when remove
                target = target.right
                
            temp = target.value
            self.remove(target.value)     #not actually removing the node, just swapping the values
            current.value = temp
        
   

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    lst = [] #list to put the contents of english.txt file into like they tell us to in the problem
    
    build_times_1 = []    #lists to store the build times in so can plot them later
    build_times_2 = []
    build_times_3 = []
    
    search_times_1 = []   #lists to store the search times in so can plot them later
    search_times_2 = []
    search_times_3 = []
    
    domain = []           #list that will put the domain values in so that can plot the build, search times over this domain of n values we want
    
    with open("english.txt", "r") as newFile:   #read the file english.txt, storing the contents of each line in list
        for x in newFile.readlines():   
            lst.append(x)                       #put the contents of each line into the list
            
    for i in range(3, 11):
        n = 2 ** i                              #loop through n from 2^3 to 2^10
        domain.append(n)                        #add the n values to the domain list so that can plot subgraphs over these n values
        
        subsets = np.random.choice(lst, n, replace = False)   #get a subset of n random items from the data set. Put replace = False so that have no duplicates in subset
        
        singly = SinglyLinkedList()             #so that we have stuff to load it in and will be recreated each time we iterate through it
        BSTtree = BST()                         #initialize objects from each of the classes so that can add the n items to them from english.txt file and plot them
        AVLtree = AVL()

        #time all of the build times that we need
        start = time.time()                    #time for singly linked list class
        for item in subsets:                   #need for loop so that can add the items to the list one at a time from the subset. otherwise will add the whole subset list
            singly.append(item)                #into the tree/linked list at once and not add them like a list/tree would. If didn't have this for loop, then would add the list
        end = time.time() - start              #as one node in the tree, etc.
        build_times_1.append(end)              #add the time to the singly linked list list so that can plot it
        
        start = time.time()                    #time for the BST class
        for item in subsets:
            BSTtree.insert(item)
        end = time.time() - start
        build_times_2.append(end)
        
        start = time.time()                    #time for AVL class
        for item in subsets:
            AVLtree.insert(item)
        end = time.time() - start
        build_times_3.append(end)
        
        #time all of the search times that we need
        random_items = np.random.choice(subsets, 5, replace = False)   #want to time for finding 5 random items that come from the subset
        
        start = time.time()                   #time the singly linked list class
        for item in random_items:
            singly.iterative_find(item)       #use this iterative_find function to find the item in the linked list (told to do this in lab manual)
        end = time.time() - start
        search_times_1.append(end)
        
        start = time.time()                   #time the BST class
        for item in random_items:
            BSTtree.find(item)                #use the find function to find it in the BST tree
        end = time.time() - start
        search_times_2.append(end)
        
        start = time.time()                   #time the AVL class
        for item in random_items:
            AVLtree.find(item)
        end = time.time() - start
        search_times_3.append(end)
        
    #now plot the time for building and searching on 2 seperate subplots. Will do the plotting outside of the for loop
    #want to plot the times on a loglog scale
    plt.subplot(1, 2, 1)                     #the first subplot have. Plot the building times here
    plt.loglog(domain, build_times_1, 'r-', label = "build SinglyLinkedList")   #plot each of them on the domain we specified (all of our n values that timed over in the for loop)
    plt.loglog(domain, build_times_2, 'b-', label = "build BST")
    plt.loglog(domain, build_times_3, 'g-', label = "build AVL")
    plt.xlabel("log(n) values")
    plt.ylabel("Exeuction Times on loglog scale")
    plt.title("Build Times")
    plt.legend(loc = "upper left")
    
    plt.subplot(1, 2, 2)                     #our second subplot have. Plot the searching times here
    plt.loglog(domain, search_times_1, 'r-', label = "search SinglyLinkedList") 
    plt.loglog(domain, search_times_2, 'b-', label = "search BST")
    plt.loglog(domain, search_times_3, 'g-', label = "search AVL")
    plt.xlabel("log(n) values")
    plt.ylabel("Execution Times on loglog scale")
    plt.title("Search Times")
    
    plt.suptitle("Compare Build and Search Times")
    plt.legend(loc = "upper left")
    
    plt.show()
      
        
        
        


 	
