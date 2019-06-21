import pandas as pd
from addTermNumber import convert_term_number

if __name__ == "__main__":

    data = pd.read_csv('..\\..\\data\\Grades.csv')

    class1 = data[data.course_name == 'Calculus and Analytic Geometry I'][['student_id', 'semester', 'year', 'grade']]
    class1['term_number'] = 0

    for i, row in class1.iterrows():
        class1.at[i, 'term_number'] = convert_term_number(class1.at[i, 'semester'], class1.at[i, 'year'])

    class1.to_csv('..\\data\\Generated_Pandas\\class1.csv')
