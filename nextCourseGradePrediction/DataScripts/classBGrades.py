from addTermNumber import convert_term_number
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('..\\..\\data\\Grades.csv')

    class2 = data[data.course_name == 'Calculus and Analytic Geometry II'][['student_id', 'semester', 'year', 'grade']]
    class2['term_number'] = 0

    for i, row in class2.iterrows():
        class2.at[i, 'term_number'] = convert_term_number(class2.at[i, 'semester'], class2.at[i, 'year'])

    class2.to_csv('..\\data\\Generated_Pandas\\class2.csv')
