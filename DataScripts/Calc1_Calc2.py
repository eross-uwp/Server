import pandas as pd

def convert_term_number():

    """
    for:
        pand[4] == season and pand[5] == year
            assign proper number
    """
if __name__ == "__main__":
    data = pd.read_csv('..\\Data\\Grades.csv')

    # var = data[data.course_name.isin(['Calculus and Analytic Geometry I', 'Calculus and Analytic Geometry II'])][['student_id', 'course_name', 'grade']]
    # var.to_csv('J:\\Research Tools\\Research\\Server\\Data\\Generated_Pandas\\Calc1_Calc2.csv')

    var = data[data.course_name == 'Calculus and Analytic Geometry I'][['student_id', 'semester', 'year', 'grade']]
    var.to_csv('J:\\Research Tools\\Research\\Server\\Data\\Generated_Pandas\\Calc1.csv')

    var = data[data.course_name == 'Calculus and Analytic Geometry II'][['student_id', 'semester', 'year', 'grade']]
    var.to_csv('J:\\Research Tools\\Research\\Server\\Data\\Generated_Pandas\\Calc2.csv')

    calc1 = pd.read_csv('..\\Data\\Generated_Pandas\\Calc1.csv')

    calc1['term_number'] = 0
    calc1.to_csv('J:\\Research Tools\\Research\\Server\\Data\\Generated_Pandas\\Calc1.csv')

    calc2 = pd.read_csv('..\\Data\\Generated_Pandas\\Calc2.csv')
    calc1['term_number'] = 0

    for i, row in calc2.iterrows():
        if calc1.at[i, 'semester'] == 'fall':
            if calc1.at[i, 'year'] == 2007:
                calc1.at[i, 'term_number'] = 740
            if calc1.at[i, 'year'] == 2008:
                calc1.at[i, 'term_number'] = 770
            if calc1.at[i, 'year'] == 2009:
                calc1.at[i, 'term_number'] = 800
            if calc1.at[i, 'year'] == 2010:
                calc1.at[i, 'term_number'] = 830
            if calc1.at[i, 'year'] == 2011:
                calc1.at[i, 'term_number'] = 860
            if calc1.at[i, 'year'] == 2012:
                calc1.at[i, 'term_number'] = 890
            if calc1.at[i, 'year'] == 2013:
                calc1.at[i, 'term_number'] = 920
            if calc1.at[i, 'year'] == 2014:
                calc1.at[i, 'term_number'] = 950
            if calc1.at[i, 'year'] == 2015:
                calc1.at[i, 'term_number'] = 980
            if calc1.at[i, 'year'] == 2016:
                calc1.at[i, 'term_number'] = 1010
            if calc1.at[i, 'year'] == 2017:
                calc1.at[i, 'term_number'] = 1040
            if calc1.at[i, 'year'] == 2018:
                calc1.at[i, 'term_number'] = 1070

        if calc1.at[i, 'semester'] == 'spring':
            if calc1.at[i, 'year'] == 2008:
                calc1.at[i, 'term_number'] = 750
            if calc1.at[i, 'year'] == 2009:
                calc1.at[i, 'term_number'] = 780
            if calc1.at[i, 'year'] == 2010:
                calc1.at[i, 'term_number'] = 810
            if calc1.at[i, 'year'] == 2011:
                calc1.at[i, 'term_number'] = 840
            if calc1.at[i, 'year'] == 2012:
                calc1.at[i, 'term_number'] = 870
            if calc1.at[i, 'year'] == 2013:
                calc1.at[i, 'term_number'] = 900
            if calc1.at[i, 'year'] == 2014:
                calc1.at[i, 'term_number'] = 930
            if calc1.at[i, 'year'] == 2015:
                calc1.at[i, 'term_number'] = 960
            if calc1.at[i, 'year'] == 2016:
                calc1.at[i, 'term_number'] = 990
            if calc1.at[i, 'year'] == 2017:
                calc1.at[i, 'term_number'] = 1020
            if calc1.at[i, 'year'] == 2018:
                calc1.at[i, 'term_number'] = 1050
            if calc1.at[i, 'year'] == 2019:
                calc1.at[i, 'term_number'] = 1080

        if calc1.at[i, 'semester'] == 'summer':
            if calc1.at[i, 'year'] == 2008:
                calc1.at[i, 'term_number'] = 760
            if calc1.at[i, 'year'] == 2009:
                calc1.at[i, 'term_number'] = 790
            if calc1.at[i, 'year'] == 2010:
                calc1.at[i, 'term_number'] = 820
            if calc1.at[i, 'year'] == 2011:
                calc1.at[i, 'term_number'] = 850
            if calc1.at[i, 'year'] == 2012:
                calc1.at[i, 'term_number'] = 880
            if calc1.at[i, 'year'] == 2013:
                calc1.at[i, 'term_number'] = 910
            if calc1.at[i, 'year'] == 2014:
                calc1.at[i, 'term_number'] = 940
            if calc1.at[i, 'year'] == 2015:
                calc1.at[i, 'term_number'] = 970
            if calc1.at[i, 'year'] == 2016:
                calc1.at[i, 'term_number'] = 1000
            if calc1.at[i, 'year'] == 2017:
                calc1.at[i, 'term_number'] = 1030
            if calc1.at[i, 'year'] == 2018:
                calc1.at[i, 'term_number'] = 1060

        if calc1.at[i, 'semester'] == 'winter':
            if calc1.at[i, 'year'] == 2008:
                calc1.at[i, 'term_number'] = 745
            if calc1.at[i, 'year'] == 2009:
                calc1.at[i, 'term_number'] = 775
            if calc1.at[i, 'year'] == 2010:
                calc1.at[i, 'term_number'] = 805
            if calc1.at[i, 'year'] == 2011:
                calc1.at[i, 'term_number'] = 835
            if calc1.at[i, 'year'] == 2012:
                calc1.at[i, 'term_number'] = 865
            if calc1.at[i, 'year'] == 2013:
                calc1.at[i, 'term_number'] = 895
            if calc1.at[i, 'year'] == 2014:
                calc1.at[i, 'term_number'] = 925
            if calc1.at[i, 'year'] == 2015:
                calc1.at[i, 'term_number'] = 955
            if calc1.at[i, 'year'] == 2016:
                calc1.at[i, 'term_number'] = 985
            if calc1.at[i, 'year'] == 2017:
                calc1.at[i, 'term_number'] = 1015
            if calc1.at[i, 'year'] == 2018:
                calc1.at[i, 'term_number'] = 1045
            if calc1.at[i, 'year'] == 2019:
                calc1.at[i, 'term_number'] = 1075



    calc1.to_csv('J:\\Research Tools\\Research\\Server\\Data\\Generated_Pandas\\Calc1.csv')
    calc1.to_csv('J:\\Research Tools\\Research\\Server\\Data\\Generated_Pandas\\Calc2.csv')
