import pandas as pd
import numpy as np


INDEX_MULTIPLIER = 3
INDEX_OFFSET = 2


def grade_finder():
    for i, row in courses.iterrows():
        reset_count()
        for j, tier in grades.iterrows():
            if grades.at[j, 'course_name'] == courses.at[i, 'unique_courses']:
                add_grade(grades.at[j, 'student_id'], grades.at[j, 'grade'], i)


def add_grade(id, grade, courseIndex):
    line = np.where(id == students['student_id'].values)
    gradeIndex = courseIndex * INDEX_MULTIPLIER
    gradeIndex = gradeIndex + INDEX_OFFSET + students[line[0], 'count']
    students.at[line[0], gradeIndex] = grade
    students[line[0], 'count'] = students[line[0], 'count'] + 1


def reset_count():
    for i, row in students.iterrows():
        students[i, 'count'] = 0


if __name__ == "__main__":

    courses = pd.read_csv('data\\uniqueCourses.csv')
    students = pd.read_csv('data\\studentGradesPerCourse.csv')
    grades = pd.read_csv('..\\Data\\Grades.csv')

    grade_finder()
