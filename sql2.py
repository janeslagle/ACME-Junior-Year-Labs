# solutions.py
"""Volume 1: SQL 2.
Jane Slagle
Volume 1 lab
3/28/22
"""

import sqlite3 as sql

# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    #query database for list of names of students who have B grade in any course
    #inner join (intersect) the StudentInfo and StudentGrades tables since they both have a StudentID column
    #and the StudentID's correspond to student names, so wont need to search through as much data when find the students who have B grades
    cur.execute("SELECT SI.StudentName "
                "FROM StudentInfo as SI INNER JOIN StudentGrades as SG "
                "ON SI.StudentID == SG.StudentID "
                "WHERE SG.Grade == 'B' ")
                
    query_result = cur.fetchall()    #get results from the query. this returns list of tuples of strings
    names = [tup[0] for tup in query_result]    #want return only list of strings so convert query_result 
    
    conn.close()  #close the connection to the database
    return names
    
# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    #query database for all tuples of form (Name, MajorName, Grade) where Name = students name, Grade = their grade in Calc
    #don't need to searchthrough the CourseInfo table because can use the CourseId column in StudentGrades table to specify only want calc
    #so only need to go through StudentInfo, MajorInfo and StudentGrades tables
    cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "   
                "ON SI.MajorID == MI.MajorID "
                "INNER JOIN StudentGrades as SG "
                "ON SI.StudentID == SG.StudentID "
                "WHERE CourseId == '1'")
    #in the query: left outer join StudentInfo, MajorInfo w/ student first bc then get all student names and wont lose majors if they dont
    #have major declared. join them on majorId bc thats what share in common
    #then inner join that with StudentGrades, and want join them on StudentId bc thats what share in common
    #and then make the WHERE condt be that courseId = 1 bc only want results if in calc
    
    query_result = cur.fetchall()
    conn.close()
    
    return query_result
               
# Problem 3
def prob3(db_file="students.db"):
    """Query the database for the list of the names of courses that have at
    least 5 students enrolled in them.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a course name.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    #want query database for list of names of courses that have at least 5 students enrolled in them
    #inner join student grades and course info table bc want number of times the course occurs and courseInfo has the courseId and StudentGrades
    #has the courseId, so joining these 2 tables will make it so that able to count the number of times the courseId occurs
    cur.execute("SELECT CI.CourseName "              #want return the course names
                "FROM StudentGrades as SG INNER JOIN CourseInfo AS CI "  
                "ON SG.CourseID == CI.CourseID "     #join the 2 tables on the courseId column
                "GROUP BY SG.CourseID "              #group by courseId in the StudentGrades table having count >= 5 bc want the course names
                "HAVING COUNT(*) >= 5;")             #that are being taken 5 or more times
   
    query_result = cur.fetchall()
    courses = [tup[0] for tup in query_result]    #want return only list of strings
    conn.close()
    
    return courses

# Problem 4
def prob4(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    #query for tuples of form (MajorName, N) where N = number of students in specified major
    #inner join MajorInfo, StudentInfo tables bc want number of students with that major so can inner join them on MajorId column
    #want include Null majors SO need a left outer join to join them (not an inner join) bc w/out left outer join: wldnt include null majors
    #and after do the left outer join: have all of the info need, so dont have to inner join it w/ anything else
    cur.execute("SELECT MI.MajorName, COUNT(*) AS num_students " 
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "   #left outer join studentInfo 1st so dont lose null majors
                "ON MI.MajorID == SI.MajorID "                              #want number of students in major, so join on MajorId
                "GROUP BY MI.MajorID "                                      
                "ORDER BY num_students DESC, MI.MajorName ASC;")            #want sort results in descending order by count N and then sort
                                                                            #in alphabetical order by MajorName
                
    query_result = cur.fetchall()
    conn.close()
    
    return query_result

# Problem 5
def prob5(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, MajorName) where
    the last name of the specified student begins with the letter C.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    #query for tuples of form (StudentName, MajorName) 
    cur.execute("SELECT SI.StudentName, MI.MajorName "
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "  #do left outer join w/ StudentInfo 1st so that include null majors
                "ON SI.MajorID == MI.MajorID "          
                "WHERE SI.StudentName LIKE '% C%';")                       #only want students whose last name begins with letter C
    
    query_result = cur.fetchall()
    conn.close()
    
    return query_result

# Problem 6
def prob6(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
   
    #want tuples of form (StudentName, N, GPA) where N = number of courses student in, GPA = their grade point average
    #want use subqueries to do bc will have subcases based on what their grade is in each course
    cur.execute("SELECT name, COUNT(*) as num_courses, AVG(GRADE) as GPA "
                "FROM ("
                    "SELECT SI.StudentName AS name, CASE Grade "     #says: in this case: look at the grades
                       "WHEN 'A+' THEN 4.0 "            #these are all the subcases for the grades. this changes the grade
                       "WHEN 'A' THEN 4.0 "
                       "WHEN 'A-' THEN 3.7 "
                       "WHEN 'B+' THEN 3.4 "
                       "WHEN 'B' THEN 3.0 "
                       "WHEN 'B-' THEN 2.7 "
                       "WHEN 'C+' THEN 2.4 "            #these are all the gpas for all the various grades
                       "WHEN 'C' THEN 2.0 "
                       "WHEN 'C-' THEN 1.7 "
                       "WHEN 'D+' THEN 1.4 "
                       "WHEN 'D' THEN 1.0 "
                       "WHEN 'D-' THEN 0.7 "
                       "ELSE 0 END AS grade "
                   "FROM StudentInfo AS SI INNER JOIN StudentGrades AS SG "    #join studentinfo, studentgrades table bc want num students in course
                   "ON SG.StudentID == SI.StudentID) "                         #want num students in courses
               "GROUP BY name "
               "ORDER BY GPA DESC ")                                           #want order results from greatest GPA to lowest
               
    query_result = cur.fetchall()
    conn.close()
    
    return query_result
        
