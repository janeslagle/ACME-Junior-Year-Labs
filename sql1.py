# sql1.py
"""Volume 1: SQL 1 (Introduction).
Jane Slagle
Vol 1 lab
3/21/22
"""

import sqlite3 as sql
import csv
from matplotlib import pyplot as plt
import numpy as np

# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    #sql.connect() creates connection to database file inputted to it (will also create one if doesn't exist already)
    #and use with to open the database file:
    with sql.connect(db_file) as conn:
        cur = conn.cursor()    #need create cursor so that able to actually work with the database info now
        
        #drop tables from database:
        cur.execute("DROP TABLE IF EXISTS MajorInfo")    #use execute() command to alter data
        cur.execute("DROP TABLE IF EXISTS CourseInfo")   
        cur.execute("DROP TABLE IF EXISTS StudentInfo")
        cur.execute("DROP TABLE IF EXISTS StudentGrades")  
        
        #add tables to database:
        cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT)")
        cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT)")
        cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER)")
        cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT)")
        
        #prob 2 stuff: fill all tables with info
        #add info to MajorInfo table first: create all rows at once, then add all rows to table at once
        major_rows = [(1, 'Math'), (2, 'Science'), (3, 'Writing'), (4, 'Art')]
        cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", major_rows)
        
        #add info to CourseInfo table:
        course_rows = [(1, 'Calculus'), (2, 'English'), (3, 'Pottery'), (4, 'History')]
        cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", course_rows)
        
        #add info to StudentInfo table:
        #firsts get the info need from the csv file
        with open("student_info.csv", 'r') as infile:
            stud_rows = list(csv.reader(infile))
        cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", stud_rows)
                
        #add info to StudentGrades table:
        with open("student_grades.csv", 'r') as filee:
            grade_rows = list(csv.reader(filee))
        cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", grade_rows)
        
        #prob 4 stuff: change all values of -1 in MajorID column with NULL values
        cur.execute("UPDATE StudentInfo SET MajorID = NULL WHERE MajorID == -1")
    
    #close database:  
    conn.close()    
    
# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    #first need to open the database:
    with sql.connect(db_file) as conn:
        cur = conn.cursor()     #need create cursor so that able to actually work with database
        
        cur.execute("DROP TABLE IF EXISTS USEarthquakes")    #drop table USEarthquakes if already exists
        cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")  #create new table 
        
        #fill table just created with data from earthquake csv file given
        with open("us_earthquakes.csv", 'r') as infile:
            rows = list(csv.reader(infile))
        cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)
        
        #prob 4 stuff:
        #remove rows from USEarthquakes table that have value of 0 for Magnitude:
        cur.execute("DELETE FROM USEarthquakes WHERE Magnitude == 0")
        
        #replace 0 values in Day, Hour, Minute, Second columns with NULL values:
        cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day==0")
        cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour==0")
        cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute==0")
        cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second==0")
       
    #close database:  
    conn.close()    
       
# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    with sql.connect(db_file) as conn:
        cur = conn.cursor()  
        
        #want all tuples of form (StudentName, CourseName) where student has A or A+ grade in that course
        #need use WHERE with SELECT since want info from different tables in database
        cur.execute("SELECT SI.StudentName, CI.CourseName "
                    "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "  #get all tables from data will need access in condts
                    "WHERE CI.CourseID == SG.CourseID AND SI.StudentID == SG.StudentID AND (SG.Grade == 'A' OR SG.GRADE == 'A+');")
        
        #get list of all tuples:
        query_result = cur.fetchall()
        
    conn.close()
    
    return query_result

# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor() 
        
    #want query USEarthquakes table for magnitudes of earthquakes during 19th century (1800-1899)
    cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year BETWEEN 1800 AND 1899")
    nin_mags = np.ravel(cur.fetchall())
        
    #want magnitudes of the earthquakes during the 20th century (1900–1999)
    cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year BETWEEN 1900 and 1999")
    twe_mags = np.ravel(cur.fetchall()) 
                    
    #find average magnitude of all earthquakes in database
    cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes;")
    avg_mags = float(np.ravel(cur.fetchall()))
    
    conn.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle("Earthquake Magnitudes")
    plt.title('19th century')
    n, bins, patches = ax1.hist(x=nin_mags,
                            alpha=0.7, rwidth=0.85, color = "darkorchid")
    ax1.grid(axis='y', alpha=0.75)
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Number of earthquakes')
    
    plt.title('20th century')
    n, bins, patches = ax2.hist(x=twe_mags,
                            alpha=0.7, rwidth=0.85, color = "mediumblue")
    ax2.grid(axis='y', alpha=0.75)
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Number of earthquakes')
    
    plt.tight_layout()
    plt.show()
    return avg_mags
    
def testing():
    #test prob 1:
    student_db(db_file="students.db", student_info="student_info.csv", student_grades="student_grades.csv") #need 1st run prob 1
    
    #then test it:
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM StudentInfo;")
        print([d[0] for d in cur.description])
        
    #test prob 2:
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM StudentInfo;"):
            print(row)
            
    #test prob 3:
    earthquakes_db()  #need 1st run prob 3
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM USEarthquakes;"):
            print(row)
             
