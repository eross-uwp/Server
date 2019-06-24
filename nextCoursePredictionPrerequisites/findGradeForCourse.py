import pandas as pd
import numpy as np


INDEX_MULTIPLIER = 3
INDEX_OFFSET = 2


def grade_finder():
    count = 0
    for i, row in courses.iterrows():
        reset_count()
        print(count)
        count = count + 1
        for j, tier in grades.iterrows():
            if grades.at[j, 'course_name'] == courses.at[i, 'unique_courses']:
                add_grade(grades.at[j, 'student_id'], grades.at[j, 'grade'], i)


def add_grade(id, grade, course_index):
    index = students.loc[students['student_id'] == id].index[0]
    grade_index = course_index * INDEX_MULTIPLIER
    grade_index = grade_index + INDEX_OFFSET + students.at[index, 'count']
    students[students.columns[grade_index + 1]] = grade
    students.at[index, 'count'] = students.at[index, 'count'] + 1


def reset_count():
    students['count'] = 0


if __name__ == "__main__":

    courses = pd.read_csv('data\\uniqueCourses.csv')
    students = pd.read_csv('data\\studentGradesPerCourse.csv')
    grades = pd.read_csv('..\\Data\\Grades.csv')

    grade_finder()
    students.to_csv('data\\studentGradesPerCourse.csv')
    print("Done")
