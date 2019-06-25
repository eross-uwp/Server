"""
___authors___: Evan Majerus
"""

import pandas as pd


INDEX_MULTIPLIER = 3
INDEX_OFFSET = 2
COURSE = 'course_name'
UNIQUE_COURSE = 'unique_courses'
STUDENT_ID = 'student_id'
GRADE = 'grade'
RETAKE_COUNT = 'count'


def grade_finder():
    count = 0
    for i, row in courses.iterrows():
        print(count)
        count = count + 1
        if count == 100:
            students.to_csv('data\\studentGradesPerCourse.csv')
        for j, tier in grades.iterrows():
            if grades.at[j, COURSE] == courses.at[i, UNIQUE_COURSE]:
                add_grade(grades.at[j, STUDENT_ID], grades.at[j, GRADE], i)


def add_grade(id, grade, course_index):
    index = students.loc[students[STUDENT_ID] == id].index[0]
    grade_index = course_index * INDEX_MULTIPLIER
    grade_index = grade_index + INDEX_OFFSET + students.at[index, RETAKE_COUNT]
    students[students.columns[grade_index]][index] = grade
    students.at[index, RETAKE_COUNT] = students.at[index, RETAKE_COUNT] + 1


def reset_count():
    students[RETAKE_COUNT] = 0


if __name__ == "__main__":

    courses = pd.read_csv('data\\uniqueCourses.csv')
    students = pd.read_csv('data\\studentGradesPerCourse.csv')
    grades = pd.read_csv('..\\Data\\Grades.csv')

    grade_finder()
    students.to_csv('data\\studentGradesPerCourse.csv')
    print("Done")
