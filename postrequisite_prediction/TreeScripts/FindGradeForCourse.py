"""
___authors___: Evan Majerus
Makes a csv of the grades that each student has gotten in every single course that they have taken including all of
their retakes.
"""

import pandas as pd


COURSE = 'course_name'
UNIQUE_COURSE = 'unique_courses'
START_COLUMN = 'starting_column'
STUDENT_ID = 'student_id'
GRADE = 'grade'
RETAKE_COUNT = 'count'

"""
Finds each time a grade occurs in the grades file and calls add_grade.
Parameters: None
Returns: None
"""
def grade_finder():
    count = 0
    for i, row in courses.iterrows():
        print(count)
        count = count + 1
        if (count % 100) == 0:
            students.to_csv('data\\studentGradesPerCourse.csv', index=False)
        reset_count()
        for j, tier in grades.iterrows():
            if grades.at[j, COURSE] == courses.at[i, UNIQUE_COURSE]:
                add_grade(grades.at[j, STUDENT_ID], grades.at[j, GRADE], courses.at[i, START_COLUMN])

"""
Adds the grade found in the grade_finder to the data frame under the correct class.
Parameters: id, grade, course_column
Returns: None
"""
def add_grade(id, grade, course_column):
    index = students.loc[students[STUDENT_ID] == id].index[0]
    grade_index = course_column + students.at[index, RETAKE_COUNT] + 1
    students.loc[index, students.columns[grade_index]] = grade
    students.at[index, RETAKE_COUNT] = students.at[index, RETAKE_COUNT] + 1

"""
resets the count that keeps track of how many times it has found a grade for each class that each student has taken.
Parameters: None
Returns: None
"""
def reset_count():
    students[RETAKE_COUNT] = 0


if __name__ == "__main__":

    courses = pd.read_csv('data\\uniqueCourses.csv')
    students = pd.read_csv('data\\studentGradesPerCourse.csv')
    grades = pd.read_csv('..\\Data\\Grades.csv')

    grade_finder()
    students.to_csv('data\\studentGradesPerCourse.csv', index=False)
    print("Done")
