import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('..\\Data\\Grades.csv')

    var = data[data.course_name.isin(['Calculus and Analytic Geometry I', 'Calculus and Analytic Geometry II'])][['student_id', 'course_name', 'grade']]
    var.to_csv('J:\\Research Tools\\Research\\Server\\Data\\Generated_Pandas\\Calc1_Calc2.csv')
