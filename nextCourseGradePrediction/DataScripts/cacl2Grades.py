from addTermNumber import convert_term_number
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('..\\..\\data\\Grades.csv')

    calc2 = data[data.course_name == 'Calculus and Analytic Geometry II'][['student_id', 'semester', 'year', 'grade']]
    calc2['term_number'] = 0

    for i, row in calc2.iterrows():
        calc2.at[i, 'term_number'] = convert_term_number(calc2.at[i, 'semester'], calc2.at[i, 'year'])

    calc2.to_csv('..\\data\\Generated_Pandas\\Calc2.csv')
