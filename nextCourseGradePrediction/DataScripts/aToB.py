"""
__Author__: Nate Braukhoff
"""

import pandas as pd
from addTermNumber import convert_term_number


def fill_class_a(sl, a):
    """
    Fills in the student's column for ClassA1, ClassA2, and ClassA3.
    Student can only retake a class a max of three time without any special permission from the school.
    :param sl: List of the student's information for the class
    :param a: DataFrame of all the grade for ClassA. (Student_id, semester, year, grade)
    :return: The DataFrame that has the columns for ClassA filled in for each student.
    """
    s = sl[0] # get the student's ID

    for i, row in a.iterrows():
        if s == a.at[i, 'student_id']:
            sl.append(a.at[i, 'grade'])

    # If a student only took that class once, the other columns still need to be filled in.
    # x = they didn't take the class for attempt n.
    while 4 > len(sl) > 1:
        sl.append('x')


def fill_class_b(sl, b):
    """
    Fills in the student's column for ClassB. We are only considering the students grade for the first attempt for
    CalssB. The grade for the second attempt will not be stored in this DataFrame.
    :param sl: List of the student's information for the class.
    :param b: DataFrame of all the grades for ClassB.
    :return: The DataFrame that has the columns for ClassB filled in for each student.
    """
    s = sl[0]  # get the student's ID

    for i, row in b.iterrows():
        if s == b.at[i, 'student_id']:
            sl.append(b.at[i, 'grade'])
            break

    # If the student didn't take classB after taking classA still need to fill in the column. If they didn't take
    # ClassB column will = x
    if len(sl) < 5:
        sl.append('x')


def get_class_grades(class_name, data):
    """
    Searches through grades.csv and will make a DataFrame that has all the classes that matches the class_name
    :param class_name: The name of the class that is going to be searched for
    :param data: grades.csv
    :return: A DataFrame with all the entries for the class.
    """
    grade_list = data[data.course_name == class_name][['student_id', 'semester', 'year', 'grade']]
    grade_list['term_number'] = 0

    for i, row in grade_list.iterrows():
        # add the term number for each row
        grade_list.at[i, 'term_number'] = convert_term_number(grade_list.at[i, 'semester'], grade_list.at[i, 'year'])

    return grade_list


def back_to_a(a, b):
    """
    Removes rows from the ClassA list when a student has taken ClassB before ClassA
    :param a: DataFrame of ClassA information
    :param b: DataFrame of ClassB information
    :return: Updated DataFrame of ClassA's information. Note that the updated DataFrame should not be bigger than the
             Original.
    """
    for i, row in b.iterrows():
        s = b.at[i, 'student_id']
        term_number = b.at[i, 'term_number']

        for j, jrow in a.iterrows():
            if s == a.at[j, 'student_id'] and term_number <= a.at[j, 'term_number']:
                a = a.drop(j)

    return a


def create_table(list_of_ids):
    """
    Creates a DataFrame for ClassA and ClassB. Then it will write to a csv file.
    :param list_of_ids: list of student's ID's.
    """
    ab = pd.DataFrame()

    for student in list_of_ids:
        # this list will have the student's grades for ClassA and ClassB. The first element of the list will contain
        # the student's ID.
        grade_list = [student]

        fill_class_a(grade_list, class_a_dataFrame)      # fill columns for ClassA
        if len(grade_list) == 4:
            fill_class_b(grade_list, class_b_dataFrame)  # fill columns for ClassB

            ab = ab.append({'student_id':grade_list[0], 'classA1': grade_list[1], 'classA2': grade_list[2],
                            'classA3': grade_list[3], 'classB': grade_list[4]}, ignore_index=True)

    ab.to_csv('..\\data\\Generated_Pandas\\aToB.csv')


if __name__ == "__main__":
    class_a_name = 'Calculus and Analytic Geometry I'
    class_b_name = 'Calculus and Analytic Geometry II'

    student_grades = pd.read_csv('..\\..\\data\\Grades.csv')

    class_a_dataFrame = get_class_grades(class_a_name, student_grades)
    class_b_dataFrame = get_class_grades(class_b_name, student_grades)

    class_a_dataFrame = back_to_a(class_a_dataFrame, class_b_dataFrame)

    id_list = class_b_dataFrame.student_id.unique()

    create_table(id_list)
