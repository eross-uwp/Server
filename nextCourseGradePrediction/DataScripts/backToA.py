import pandas as pd


def back_to_a(a, b):

    for i, row in b.iterrows():
        student = b.at[i, 'student_id']
        term_number = b.at[i, 'term_number']

        for j, jrow in a.iterrows():
            if student == a.at[j, 'student_id'] and term_number <= a.at[j, 'term_number']:
                a = a.drop(j)

    a.to_csv('..\\data\\Generated_Pandas\\classA.csv')