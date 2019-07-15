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

# Finds each time when a grades occurs in the grades file of for each course for each student.
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

# Adds the grade from the final data sheet to the dataframe under the correct student and course.
def add_grade(id, grade, course_column):
    index = students.loc[students[STUDENT_ID] == id].index[0]
    grade_index = course_column + students.at[index, RETAKE_COUNT] + 1
    students.loc[index, students.columns[grade_index]] = grade
    students.at[index, RETAKE_COUNT] = students.at[index, RETAKE_COUNT] + 1

# Resets the retake counter for each student in the dataframe.
def reset_count():
    students[RETAKE_COUNT] = 0


if __name__ == "__main__":

    courses = pd.read_csv('data\\uniqueCourses.csv')
    students = pd.read_csv('data\\studentGradesPerCourse.csv')
    grades = pd.read_csv('..\\Data\\Grades.csv')

    grade_finder()
    students.to_csv('data\\studentGradesPerCourse.csv', index=False)
    print("Done")
