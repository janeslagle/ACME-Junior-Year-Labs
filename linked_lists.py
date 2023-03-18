# linked_lists.py
"""Volume 2: Linked Lists.
Jane Slagle
Math 321 Section 2
10/3/21
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute. Only accepts data of type int, float, or str.
        Raises a TypeError with an appropriate error message if another type of data is given.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        
        if type(data) != int and type(data) != float and type(data) != str:    #check if data is of type int, float, or str
            raise TypeError("Data is not of type int, float, or str")          #if data is not one of those given types, then raise the TypeError
        else:                                                                  #if data is one of the appropriate types, then store data in value attribute
            self.value = data
          

class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.counter = 0    #create attribute to track current size of list

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
            
        self.counter = self.counter + 1    #update counter every time append something

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        
        #given where in the list it is located and we have to then find it in the list and return the location 
        #in the linked list: head is the first thing, tail is the last. each node is pointing to the next thing (the things its connected to)
        #find the location of data in whole linked list
        
        if self.head is None:     #check if list is empty before start
            raise ValueError("The list is empty")
       
        current_node = self.head
        #start at the head:
        while current_node != None:  #if self.head equals none then gone through the whole list. start at beginning of linkedlist
            if current_node.value == data:    #check if the 1st thing in linked list equals data
                return current_node           #if it does, then return it because that's what we want
            
            current_node = current_node.next     #update the value of self.head because if its not equal to data, then want move on to next value in linked list 
                                           #before go through while loop again
            
        if current_node is None:     #if get through while loop and self.head still equals none, then there is no node thats equal to data, so check again 
            raise ValueError("No such node exists")
        

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        
        if i < 0:      
            raise ValueError("i cannot be negative")
        if i >= self.counter:    #counter tells you how long list is
            raise IndexError("i cannot be greater than or equal to number of nodes in list")
        
        node_current = self.head    
        while i > 0:                #if i is 0, never go through list. start at head and then start loop             
            node_current = node_current.next   #move on to the next node in list
            i = i - 1               #if i is 1, then returns the one after head like want. indexing starts at 0                          
        return node_current         #whatever node_current is at end of while loop, will be ith element because loop through it i times
       

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.counter   #counter tells you how long list is
        
        
    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        
        string = "["
        
        current_node = self.head
        
        #move through the list, put each node value in string
        
        for i in range(self.counter):   #loop through the whole list, using self.counter because it tells you how many things are in list
            if i != self.counter - 1:   #dont add it if its the last thing because need closing bracket on last one, so stop at second to last one
                if type(current_node.value) == str:
                    string = string + '\'' + current_node.value + '\', '
                else:
                    string = string + str(current_node.value) + ", "
            else:  #for the last element
                if type(current_node.value) == str:
                    string = string + "\'" + current_node.value + "\'"
                else:
                    string = string + str(current_node.value)
                    
            current_node = current_node.next
            
        string = string + "]"
        return string
                    
                    
    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        target = self.find(data)           #use find() method from problem 2 to locate the target node want to remove
                                           #calling self.find will check if the list is empty or no such node exists, so don't have
                                           #call another ValueError to find these things
        
        if self.counter == 1:              #check if only have 1 thing in the list
            self.head = None               #makes an empty list because the head and tail point to nothing
            self.tail = None
        elif self.head.value == data:      #check if 1st node in list is one want to remove
            self.head.next.prev = None     #1st node won't have previous node before it, so set to None
            self.head = self.head.next     #reassign the head to be at the new start of the list
        elif self.tail.value == data:      #check if last node in list is one want to remove
            self.tail.prev.next = None     #last node won't have next node after it, so set the next one to be None
            self.tail = self.tail.prev     #reassign the tail so to be at the new spot that is end of list
        else:
        #the next two lines are for removing all of the elements except the first, last nodes in the list
            target.prev.next = target.next     #want this value to be the node after our target node
            target.next.prev = target.prev     #want this value to be the node before our target node
       
        self.counter = self.counter - 1    #need to decrement your counter here

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        
        if index < 0:             #raise the errors depending on what index value is
            raise IndexError("Index cannot be negative")
        if index > self.counter:
            raise IndexError("Index cannot be greater than number of nodes in list")
            
        new_node = LinkedListNode(data)   #creates new node for data
            
        if index == 0:   #special case when need add before first element in list
            self.head.prev = new_node   #create the node before the head node and assign new_node value to it
            new_node.next = self.head   #put the original self.head in the second node spot now (pushing it down list by 1)
            self.head = new_node        #assign the new self.head (the new node adding) the new node value
            self.counter = self.counter + 1   #need increment number of nodes in list
        elif index == self.counter:  #if index is equal to number of nodes in list
            self.append(data)        #append node to the end of list by calling append()
        else:                        #for when inserting nodes that aren't inserting at start, end of list
            index_replacing = self.get(index)   #use get() method to get the actual node at the given index
            index_replacing.prev.next = new_node    #create new spot between index want to replace and one before it and put the new node there
            new_node.next = index_replacing         #put the original thing that was in the index originally after the new node just inserted
            self.counter = self.counter + 1        #need to increment the number of nodes each time add something
        

# Problem 6: Deque class.
class Deque(LinkedList):
    """Deque class that inherits from LinkedList class."""
    
    def __init__(self):
        LinkedList.__init__(self)   #automatically get everything from LinkedList constructor because want all the same stuff

    def pop(self):
        """Remove the last value from the list."""
        if self.head is None:     #check if list is empty
            raise ValueError("The list is empty")
        else:
            last_node = self.get(self.counter - 1)       #want remove last node, self.counter is #nodes have, so this gives you index last node 
            LinkedList.remove(self, last_node.value)  #have use LinkedList because want use remove function that wrote for other class to remove it from list 
                                                      #do .value because want pass in value to remove, not the actual node 
                                          #don't want to call the remove function in this deque class  
            return last_node.value
    
    def popleft(self):
        """Remove the first value from the list."""
        if self.head is None:      #check if list is empty
            raise ValueError("The list is empty")
        else:
            first_node = self.get(0)     #gets first node in the deque
            LinkedList.remove(self, first_node.value)
            
            return first_node.value
            
    def appendleft(self, value):
        """Insert a new node at the beginning of the list."""
        
        LinkedList.insert(self, 0, value)   #want to insert at 0th index (beginning of list) and insert value they input
        
            
    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")
        
    def insert(*args, **kwargs):
        raise NotImplementedError("Use append() or appendleft() for inserting")


# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    D = Deque()    #initialize a deque using the Deque class from above
    
    with open(infile, "r") as newFile:   #read the file, "r" means you are reading the file. Call new file newFile
        for x in newFile.readlines():    #loop through all lines in file, saying for each line 
            D.append(x)                  #add each line to deque D
    #now have read the entire file         
           
    with open(outfile, "w") as output_File:  #write to the second file which is outfile, but access outfile as output_File 
        for x in range(len(D)):              #loop through every entry have in Deque D, popping each entry x in deque off one at a time
            output_File.write(D.pop())       #use deque function pop() to pop each entry off the dock one at a time since it's in for loop
            
  
           
