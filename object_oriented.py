# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Jane Slagle
Math 321 section 2
9/7/21
"""

import math #needed for problem 4

class Backpack:
    """A Backpack object class. Has a name, color, maximum amount of items you can put into it and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color (str): the color of the backpack
        max_size (int): the amount of items in the contents list of the backpack
        
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name, initialize an empty list of contents, initialize and set the maximum number of items in the backpack and set the 
        color of the backpack.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size (int): the max number of items in the backpack
        """
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        """Add an item to the backpack's list of contents."""
        if len(self.contents) >= self.max_size: #check if amount of items (given by length of contents list because all items we have are in contents list) is >= max_size value set above
            print("No room!")                  #if already have more than max_size items in contents list, do not add item
        else:
            self.contents.append(item)    #if there are not already max_size items in contents list, add the item

    def dump(self):
        """Resets the contents of the backpack to an empty list."""
        self.contents = []

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

    def __eq__(self, other):  #pass in first object as self, pass in second object as other
        """Used to determine if two objects in the Backpack class are equal. Returns
        true if name, color and number of contents of the two objects are equal. Otherwise,
        return False."""
        
        if self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents):  #check if the two objects have the same name, color and number of contents
            return True 
        else:
            return False
        
    def __str__(self):  #only need one object because use this on one object at a time
        """Returns the string representation of an object."""
        
        string_1 = "Owner:" + '\t\t' + self.name +'\n'        
        string_2 = "Color:" + '\t\t' + self.color + '\n'           
        string_3 = "Size:" + '\t\t' + str(len(self.contents)) + '\n'   #need to use str() with anything that is not a string
        string_4 = "Max Size:" + '\t' + str(self.max_size) + '\n'       
        string_5 = "Contents:" + '\t' + str(self.contents)   
        
        return string_1 + string_2 + string_3 + string_4 + string_5   #return all of those strings together by adding them all
                
# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)

# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class. A jetpack is a backpack that allows you to fly when you wear it.
    
    Attributes:
        name (str): the name of the jetpack's owner.
        color (str): the color of the jetpack.
        max_size (int): the maximum number of items that can fit inside the jetpack.
        contents (list): the contents of the jetpack.
        fuel (int): the amount of fuel in the jetpack.
    """
    
    def __init__(self, name, color, max_size = 2, fuel = 10):
        """Use the Backpack constructor to initalize the name, color, and max_size attributes. A jetpack can only hold 2 items by default
        and the amount of fuel is 10 by default.
        
        Parameters:
            name (str): the name of the jetpack's owner.
            color (str): the color of the jetpack.
            max_size (int): the maximum number of items that can fit inside the jetpack.
            fuel (int): the amount of fuel available in the jetpack.
        """
        Backpack.__init__(self, name, color, max_size)  #use the Backpack constructor to initalize name, color and max_size
        self.fuel = fuel  #need to store the new fuel parameter added as an attribute

    def fly(self, fuel_burned):
        """Accepts amount of fuel to be burned and decrements the fuel attribute by that amount."""
        if fuel_burned <= self.fuel:       #if the fuel_burned is less than the amount of fuel have left
            self.fuel = self.fuel - fuel_burned    #decrement the fuel attribute by the fuel_burned amount so know how much fuel have left
        else:
            print("Not enough fuel!")  #if fuel_burned is equal to or more than the amount of fuel left then do not decrement fuel
                   
    def dump(self):
        """Override dump() from the Backpack class so that both the contents and fuel tank are emptied."""
        Backpack.dump(self)
        self.fuel = 0      

            
# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:
    """A ComplexNumber object class. Complex numbers are denoted a + bi, where a is a real number and i is an imaginary number.
    Thus, the ComplexNumber object class has a constructor that accepts 2 numbers, a real attribute and an imaginary attribute.
    The class here is getting the values of a and b.

    Attributes:
        real (int): the real part of the complex number, we will consider it to be "a" in "a + bi"
        imag (int): the imaginary part of the complex number, we will consider it to be "b" in "a + bi" since we will always have b times the 
        imaginary part so just consider b as imaginary part as a whole
      
    """
    def __init__(self, real, imag):
        """Set the real and imaginary parts of the complex number.

        Parameters:
            real (int): the real part of the complex number
            imag (int): the imaginary part of the complex number
          
        """
        self.real = real
        self.imag = imag
        
    def conjugate(self):   #this only needs self as a parameter in the function
        """Compute the conjugate of the complex number."""
        
        return ComplexNumber(self.real, self.imag * (-1))  #put ComplexNumber around it to make it a ComplexNumber object
                                                          #keep the self.real part the same because with conjugate, only the bi part is changing (so only the imaginary part)
    						            #and it changes because it is becoming negative, so multiply self.imag by -1 here to create the conjugate
    						            #since it will return self.real as a and self.imag as -bi like how we want
    
    #Magic Methods:
    
    def __str__(self):
        """Print out the complex number in the correct form depending on the value of b have."""
        if self.imag >= 0:
            real_string = str(self.real) + "+"    #create the correct real part of the string
            imag_string = str(self.imag) + "j"    #create the correct imaginary part of the string
        else:
            real_string = str(self.real) 
            imag_string = str(self.imag) + "j"
            
        return("(" + real_string + imag_string + ")")   #return the correct string based on what value of b have
        
    def __abs__(self):
        """Determines the output of the built-in abs() function (absolute value) and implements it so that it returns
        the magnitude of the complex number."""
        magnitude = math.sqrt((self.real ** 2) + (self.imag ** 2))  #use the squareroot function from the math module to get the squareroot so that can find the magnitude
        
        return magnitude
        
    def __eq__(self, other):
        """Checks if 2 ComplexNumber objects are equal. They are equal if and only if they have the same real and imaginary parts."""
        if self.real == other.real and self.imag == other.imag:      #check if the 2 objects have the same real and imaginary parts
            return True   
        else:
            return False
        
    def __add__(self, other):
        """Adds 2 ComplexNumber objects, adding their real and imaginary parts together."""
        return ComplexNumber(self.real + other.real, self.imag + other.imag)   #create and return new object that has the added real and imaginary numbers
        
    def __sub__(self, other):
        """Subtracts 2 ComplexNumber objects, subtracting their real and imaginary parts together."""
        return ComplexNumber(self.real - other.real, self.imag - other.imag)   #create and return new object that has the subtracted real and imaginary numbers
        
    def __mul__(self, other):
        """Multiplies 2 ComplexNumber objects. So, you are multiplying 2 objects, the first of the form (a+bi), the second of the form of 
        (c+di) and when you multiply them out, you get ac+adi+bci+bd(i^2) which equals (ac-bd) + (ad+bc)i. Thus, we will return 
        (ac-bd) + (ad+bc)i in this function."""
        
        real_part = (self.real * other.real) - (self.imag * other.imag)   #do computations to match what we need the real part to be in the new object
        imag_part = (self.real * other.imag) + (self.imag * other.real)   #do computations to match what we need the imaginary part to be in the new object
        
        return ComplexNumber(real_part, imag_part)
        
    def __truediv__(self, other):
        """Division of 2 ComplexNumber objects, so division of 2 complex numbers. You divide 2 complex numbers by multiplying by the
        conjugate. Then, doing this out, we find that we want to return ((a+bi)(c-di) / (c^2 + d^2)). We will use the functions that 
        we have thus made so far. So, we see that the new object we want to make will need to call on the multiplication object, etc."""
        
        numerator = (self) * (other.conjugate())        #using * here calls the multiplication magic method that wrote above and call the conjugate function as well
        denominator = (other) * (other.conjugate())    
        
        newReal = numerator.real / denominator.real  #numerator.real gets the real part out of what the multiplication function gives you. The denominator only has 
                                                     #real numbers in it (the imag part is 0), so just access the real part of the denominator
        newImag = numerator.imag / denominator.real  #do the same thing, but for the imaginary part from the multiplication function now
        
        return ComplexNumber(newReal, newImag)  #create the new complex number object by using the real and imaginary parts we specified above


"""def test_backpack():                         #used to test the Backpack class
    myBackpack = Backpack("Jane", "red")
    if myBackpack.name != "Jane":
        print("Name is wrong")
    for item in ("textbook", "pen", "phone"):
        myBackpack.put(item)
    print("Contents: ", myBackpack.contents)"""
     
"""def test_backpack_magic_methods():           #used to test the magic methods in Backpack class
    myBackpack = Backpack("JANE", "purple")
    secondBackpack = Backpack("Kimberly", "green")
    thirdBackpack = Backpack("JANE", "purple")
    print(str(myBackpack))
    print(myBackpack == secondBackpack)
    print(myBackpack == thirdBackpack)"""
    
"""def test_jetpack():                             #used to test the JetPack class
    myJetpack = Jetpack("Jane", "purple")
    print(myJetpack)
    
    myJetpack.fly(5)
    print(myJetpack.fuel)
    myJetpack.fly(2000)"""
    
    
"""def test_ComplexNumber(a, b):                 #used to test the ComplexNumber Class
    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)

    # Validate the constructor.
    if my_cnum.real != a or my_cnum.imag != b:
        print("__init__() set self.real and self.imag incorrectly")

    # Validate conjugate() by checking the new number's imag attribute.
    if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
        print("conjugate() failed for", py_cnum)

    # Validate __str__().
    if str(py_cnum) != str(my_cnum):
        print("__str__() failed for", py_cnum)
        
    # Validate __abs__().
    if abs(py_cnum) != abs(my_cnum):
        print("__abs__() failed for", py_cnum)
        
    #Validate __eq__().
    if py_cnum != my_cnum:
        print("__eq__() failed for", py_cnum)"""
   
"""def test_ComplexNumberArith(a, b, c, d):               #Used to test the ComplexNumber class magic methods
    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)
    second_py_cnum = complex(c, d)
    second_cnum = ComplexNumber(c, d)
    
    #Validate __add__().
    if (py_cnum + second_py_cnum) != (my_cnum + second_cnum):
        print("__add__() failed for", py_cnum)
    
    #Validate __sub__().
    if (py_cnum - second_py_cnum) != (my_cnum - second_cnum):
        print("__sub__() failed for", py_cnum)
        
    #Validate __mul__().
    if (py_cnum * second_py_cnum) != (my_cnum * second_cnum):
        print("__mul__() failed for", py_cnum)
        
    #Validate __truediv__().
    if (py_cnum / second_py_cnum) != (my_cnum / second_cnum):
        print(my_cnum / second_cnum)
        print(py_cnum / second_py_cnum)
        print((my_cnum / second_cnum) - (py_cnum / second_py_cnum))
        print("__truediv__() failed for", py_cnum)"""
        

    
        

