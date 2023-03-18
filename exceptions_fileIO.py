# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Jane Slagle
Math 321 Section 2
9/20/21
"""

from random import choice
import numpy as np         #need for problem 4

# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:
    
    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
                                     
    if len(step_1) != 3:   #check if first number (step_1) is not a 3 digit number. Use len to get length of the step_1 number
        raise ValueError("The first number was not a 3 digit number!")  #raise ValueError exception with error message if 1st number not 3 digits long
    if abs(int(step_1[0]) - int(step_1[2])) < 2:
        raise ValueError("The last digits of the first number do not differ by 2 or more.")  #raise ValueError exception if first, last digits don't differ by 2 or more
                                               
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
                                       
    if step_2 != step_1[::-1]:   #the number is a string because of the input() function, so can check the reverse of the string using slicing
        raise ValueError("The second number is not the reverse of the first number!")  #want the second number to be the reverse of the 1st number
                            
    step_3 = input("Enter the positive difference of these numbers: ")
    
    if int(step_3) != abs(int(step_1) - int(step_2)):   #check if 3rd number is the positive difference of the 1st 2 numbers. Use abs() to make sure it is the positive difference
        raise ValueError("The third number is not the positive difference of the first two numbers.")
    
    step_4 = input("Enter the reverse of the previous result: ")
    
    if step_4 != step_3[::-1]:     #check if the 4th number is the reverse of the 3rd number
        raise ValueError("The fourth number is not the reverse of the third number.") 
    
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")
    

# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the 
    program is running, the function should catch the exception and 
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    
    walk = 0
    directions = [1, -1]
    
    try:                                  #need try block so that able to find if there is a KeyboardInterrupt exception at any point inside the try block code
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:             #detects if there is a KeyBoardInterrupt exception inside the try block code (inside lines 67 - 68)
        print("Process interrupted at iteration", i)   #print this message if there is a keyboard interrupt exception. Want the i from for loop that keyboard interrupt happens at
    else:                                 #if there is no keyboard interrupt exception, it will go to this else statement instead and print that the process was completed like want
        print("Process completed")         
    finally:                              #finally executes no matter what the except and else statements are, so have this with return walk in it because in both cases, want return walk
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
    """Class for reading in file
        
    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
        
    """
class ContentFilter(object):   
    # Problem 3
    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        self.filename = filename          #store the file's name as attribute in the class
        run = True                        
        
        while run == True:                #only go back through while loop if file was invalid
            try:
                with open(filename, 'r') as myFile:  #open the file for reading, you now access the file as myFile
                    self.contents = myFile.read()    #store the file's contents as attribute in class as string. Use read() to read in content and it will be string automatically 
                    run = False                      #set run to False because if made it through whole try block without an exception then file valid
            except (FileNotFoundError, TypeError, OSError):          #raises the possible exceptions for the file name entered if the given file name is invalid in any way
                filename = input("Please enter a valid file name:")    #prompt user for another filename using input() and then will loop back through and check if it's valid       
                
        #Calculate the statistics for the string magic method in problem 4:
        
        self.numLetters = sum([char.isalpha() for char in self.contents])       #calculate the number of letters have
        self.numNums = sum([s.isdigit() for s in self.contents])                #calculate the number of digits have
        self.numWhiteSpaces = len(self.contents.split((" "))) - 1 + len(self.contents.split(("\n"))) - 1 + len(self.contents.split(("\t"))) - 1  #find number whitespaces: spaces, newlines, tabs
        self.numLines = len(self.contents.split('\n')) - 1                      #calculate the number of lines have
        self.numChars = len(self.contents)                                      #find total characters because contents is a string
    
 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""
        
        if mode != 'w' and mode != 'x' and mode != 'a':    #raise ValueError exception if the mode is not one of the correct options. Need and instead of or here because only want to
            raise ValueError("The mode is invalid")        #raise the exception if the mode is not equal to any one of these


    def uniform(self, outfile, mode='w', case='upper'):
        """Write the data to the outfile in uniform case."""
        
        self.check_mode(mode)                           #have check if have valid mode first
        
        if case == 'upper':
            with open(outfile, mode) as myFile:         #self.filename has the name of the file, and opening the file as outfile here
                myFile.write(self.contents.upper())     #self.contents will give you contents of outfile and want to write data to outfile all as upper case letters 
        elif case == 'lower':
            with open(outfile, mode) as myFile:
                myFile.write(self.contents.lower())     #self.contents will give you contents of outfile and want to write data to outfile in lower case
        else:    
            raise ValueError("Case must be upper or lower")   #raise ValueError exception if case is not upper or lower
        
    def reverse(self, outfile, mode='w', unit='line'):
        """Write the data to the outfile in reverse order."""
        
        self.check_mode(mode)                           #check to see if have a valid mode first  
        
        with open(outfile, mode) as myFile:   #open the file, access it as myFile now. The file will be open for remainder of this function, so don't need open it again if make 
                                              #rest of code for this problem be within this with statement
            lineSplit = self.contents.strip().split('\n')     #want split each line so split at each newline so it's separate lines. Do strip() to avoid having an empty list
                                                              #this creates a list where each element is each line
            if unit == 'word': 
                wordSplit = [str(word[::-1]) for word in lineSplit]  #split each word, this puts them in a list, reverse them and cast list to string, do for every word in each line
                for i in range(len(wordSplit)):                      #go through all elements in the now splitted word list
                    myFile.write(wordSplit[i])                       #add each word to the file by writing them there
                    myFile.write("\n")                               #you write each word on each line, once get here: done with the line, so have newline so goes on to the next line
            elif unit == 'line':
                newLines = lineSplit[::-1]                           #reverse all the lines in the splitted lines list made above and put them in a new list
                for i in range(len(newLines)):                       #loop through all of the now reversed splitted lines have
                    myFile.write(newLines[i])                        #write each element in the newLines list to our file
                    myFile.write("\n")                               #put newline at end so each line on a different line from one another
            else:
                raise ValueError("Unit must be word or line")            
            
    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""
        
        self.check_mode(mode)     
        
        with open(outfile, mode) as newFile:
            lines = self.contents.strip('\n').split('\n')   #use strip because otherwise will make an empty list at the end. Split each line so each line is separate in a list
            matrix = []                                     #initialize the matrix as an empty list
            for i in range(len(lines)):                     #loop through list of all the lines
                words = lines[i].split(" ")                 #go through each line and split the words in each line
                matrix.append(words)                        #add all of the words to our matrix
            matrix2 = np.array(matrix)                      #need to make an array because to use .T transpose function on something, it needs to be an array
            transposed_matrix = matrix2.T                   #transposing the matrix, so don't need transpose it in the nested for loops below
            for i in range(len(transposed_matrix)):         #loop through all lines in the matrix 
                for j in range(len(transposed_matrix[i])):  #loop through all words in each line in the matrix
                    newFile.write(transposed_matrix[i][j] + " ")     #writing to the new file making: it's already transposed, so just need add each element. 
                                                                     #loop through each line 1 line at time
                newFile.write('\n')                         #going through each word in line, adding each word. Put space so space btw each word in each line. 
                                                            #have newline so each line separate from each other. Have newline before loop back through so lines indices so that move on to
                                                            #the next line
                                                            
    def __str__(self):
        """String representation: info about the contents of the file."""
        
        string_1 = "Source file:" + '\t\t\t' + str(self.filename) + '\n'
        string_2 = "Total characters:" + '\t\t' + str(self.numChars) + '\n'
        string_3 = "Alphabetic characters:" + '\t\t' + str(self.numLetters) + '\n'
        string_4 = "Numerical characters:" + '\t\t' + str(self.numNums) + '\n'
        string_5 = "Whitespace characters:" + '\t\t' + str(self.numWhiteSpaces) + '\n'
        string_6 = "Number of lines:" + '\t\t' + str(self.numLines) + '\n'
        
        return string_1 + string_2 + string_3 + string_4 + string_5 + string_6 
        
"""if __name__ == "__main__":                             #write main function so that able to test all functions wrote in problem 4
    cf = ContentFilter("cf_example1.txt")
    #cf.uniform("uniform.txt", mode='w', case="upper")
    #cf.uniform("uniform.txt", mode='a', case="lower") 
    #cf.reverse("reverse.txt", mode='w', unit="word")   
    #cf.reverse("reverse.txt", mode='a', unit="line")   
    #cf.transpose("transpose.txt", mode='w')"""


