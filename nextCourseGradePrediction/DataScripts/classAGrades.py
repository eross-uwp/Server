import pandas as pd
from addTermNumber import convert_term_number


def get_class_a(class_name, data):
    # data = pd.read_csv('..\\..\\data\\Grades.csv')

    a = data[data.course_name == class_name][['student_id', 'semester', 'year', 'grade']]
    a['term_number'] = 0

    for i, row in a.iterrows():
        a.at[i, 'term_number'] = convert_term_number(a.at[i, 'semester'], a.at[i, 'year'])

    return a
