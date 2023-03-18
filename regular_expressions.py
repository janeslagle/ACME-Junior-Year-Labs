# regular_expressions.py
"""Volume 3: Regular Expressions.
Jane Slagle
Math 323 Volume 2 lab
2/13/22
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    pattern = re.compile("python")    #make pattern object for finding 'python'. This compiles it
    return pattern                    #return the regular expression patter object with pattern string 'python'

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    string_cheese = re.compile(r"\^\{\@\}\(\?\)\[\%\]\{\.\}\(\*\)\[\_\]\{\&\}\$")  #need to compile, return given string
    return string_cheese

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #use re.compile because want to compile it
    reggy = re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")
    #set it up like the one, two fish example in lab manual
    #want each of these strings to be matched only with each other, so use ^ and $ and the pipe |. Need a \ in front of everything
    
    return reggy
    
# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #want to make a regular expression pattern object like all the other problems and need to compile it:
    py_thing = re.compile(r"^[a-zA-Z_](\w|_){0,} {0,}(|= {0,}(\d{0,}(|\.\d{0,})|'[^']{0,}'|[a-zA-Z_](\w|_){0,}))$")
     
    return py_thing
    
# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    keywords = r"(^\s*(if|elif|else|for|while|try|except|finally|with|def|class)[^\n]*)"  #create the keywords with any number of letters
                                                                                         #after the keyword. put all keywords into group
                                                                                 #together: put them inside () together
                                                                                 #make the entire expression into 1 group together
                                                                                 #that is the outside parenthesis: and do everything
                                                                                 #except for a newline: thats what the last brackets 
                                                                                 #do. and put a star bc there are a lot of things
                                                                                 #that arent newlines
                                                                                 
    
    pattern = re.compile(keywords, re.MULTILINE)                                 #search through all of the lines at once
     
    new_code = pattern.sub(r"\1:", code)                                            #subs keyword line with colon at end into OG code
                                                                       #want to sub group 1 because group 1 is the keywords here
    return new_code
    
# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    #first read from fake_contacts.txt file:
    with open(filename, "r") as file:
        lines = file.readlines()
    
    #get string patterns for names, bdays, emails, phone numbers:
                                 #firstname      #opt mid init  #end w/ last name w/ anything possible after
    name_pattern = re.compile(r"[A-Z][a-z]*[ ][A_Z]?[.]?[ ]?[A-Z][a-z]+")
    #? means that can do it either 0 or 1 time: references whatever precedes ?. + means 1 or more (so doesn't include 0) - think it's same as *
                                 #month  #day    #year
    bday_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")
    #put everything in brackets because that says that it needs to match it 
    email_pattern = re.compile(r"[\w]*[.]?[\w]*[.]?[\w]*[.]?[\w]*[.]?[@][\w]*[.]?[\w]*[.]?[\w]*[.]?[\w]*[.]?")
    phone_pattern = re.compile(r"[\d]?[-]?[(]?[\d]+[)]?[-]?[\d]+[-]?[\d]{4}")
             
    dict = {}             #main overall dictionary
    for line in lines:    #loop through each line in file
        #format birthdays uniformly as mm/dd/yyyy:
        if bool(bday_pattern.search(line)) == True:    #check if even a birthday
            bday = bday_pattern.findall(line)          #change each line in our birthday list
            bday = bday[0]
            bday = str(bday)
            if bday[2] != "/":                         #means there is no 0 in the first spot of month
                bday = "0" + bday[0:]                  #add 0 to month, add everything else on
            elif bday[5] != "/":                       #add first 0 to day
                bday = bday[:3] + "0" + bday[3:]
            elif len(bday) != 10:                      #make year 4 digits
                bday = bday[:6] + "20" + bday[6:]
        if bool(bday_pattern.search(line)) == False:
            bday = None
        
        #format phone numbers as (xxx)xxx-xxxx:
        if bool(phone_pattern.search(line)) == True:
            phone = phone_pattern.findall(line)
            phone = phone[0]
            phone = str(phone)
            if phone[1] == "-":
                phone = phone[2:]
            if phone[0] != "(":
                phone = "(" + phone[:3] + ")" + phone[3:]
            if phone[5] == "-":
                phone = phone[0:5] + phone[6:]   
        if bool(phone_pattern.search(line)) == False:
            phone = None
        
        #get names, emails:
        if bool(name_pattern.search(line)) == True:    #check if even a birthday
            name = name_pattern.findall(line)          #change each line in our birthday list
            name = name[0]
            name = str(name)
        if bool(email_pattern.search(line)) == True:
            email = email_pattern.findall(line)
            email = email[0]
            email = str(email)
        if bool(email_pattern.search(line)) == False:
            email = None
            
        #get the dictionary:
        dict[name] = {"birthday": bday, "email": email, "phone": phone}    #want key of the dictionary to be the names
    
    return dict

