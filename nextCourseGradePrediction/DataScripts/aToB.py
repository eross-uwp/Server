import pandas as pd
from addTermNumber import convert_term_number


def fill_class_a(sl, a):
    s = sl[0]

    for i, row in a.iterrows():
        if s == a.at[i, 'student_id']:
            sl.append(a.at[i, 'grade'])

    while 4 > len(sl) > 1:
        sl.append('x')


def fill_class_b(sl, b):
    s = sl[0]

    for i, row in b.iterrows():
        if s == b.at[i, 'student_id']:
            sl.append(b.at[i, 'grade'])
            break
    if len(sl) < 5:
        sl.append('x')


def get_class_a(class_name, data):
    a = data[data.course_name == class_name][['student_id', 'semester', 'year', 'grade']]
    a['term_number'] = 0

    for i, row in a.iterrows():
        a.at[i, 'term_number'] = convert_term_number(a.at[i, 'semester'], a.at[i, 'year'])

    return a


def get_class_b(class_name, data):
    b = data[data.course_name == class_name][['student_id', 'semester', 'year', 'grade']]
    b['term_number'] = 0

    for i, row in b.iterrows():
        b.at[i, 'term_number'] = convert_term_number(b.at[i, 'semester'], b.at[i, 'year'])

    return b


def back_to_a(a, b):
    for i, row in b.iterrows():
        s = b.at[i, 'student_id']
        term_number = b.at[i, 'term_number']

        for j, jrow in a.iterrows():
            if s == a.at[j, 'student_id'] and term_number <= a.at[j, 'term_number']:
                a = a.drop(j)

    return a


def create_table(l):
    ab = pd.DataFrame()

    for student in l:
        grade_list = [student]

        fill_class_a(grade_list, class_a_dataFrame)
        if len(grade_list) == 4:
            fill_class_b(grade_list, class_b_dataFrame)

            ab = ab.append({'student_id':grade_list[0], 'classA1': grade_list[1], 'classA2': grade_list[2],
                            'classA3': grade_list[3], 'classB': grade_list[4]}, ignore_index=True)

    ab.to_csv('..\\data\\Generated_Pandas\\aToB.csv')


if __name__ == "__main__":
    class_a_name = 'Calculus and Analytic Geometry I'
    class_b_name = 'Calculus and Analytic Geometry II'

    student_grades = pd.read_csv('..\\..\\data\\Grades.csv')

    class_a_dataFrame = get_class_a(class_a_name, student_grades)
    class_b_dataFrame = get_class_b(class_b_name, student_grades)

    class_a_dataFrame = back_to_a(class_a_dataFrame, class_b_dataFrame)

    id_list = class_b_dataFrame.student_id.unique()

    create_table(id_list)
