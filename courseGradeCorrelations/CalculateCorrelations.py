from scipy.stats import spearmanr
import pandas as pd

COURSE_LIST = pd.read_csv('data\\raw_courses.csv')['courses']
STUDENT_ID_LIST = pd.read_csv('data\\raw_student_ids.csv')['student_id']
GRADES_LIST = pd.read_csv('data\\raw_grades.csv')
GRADES_LIST[['student_id', 'term_number']] = GRADES_LIST[['student_id', 'term_number']].apply(pd.to_numeric)
STUDENT_GRADE_LIST = pd.read_csv('data\\student_grade_list.csv')
COURSE_COMBINATIONS = pd.read_csv('data\\course_combinations.csv')

'''
For each row in the course combinations table, get class 1 and class 2. Create a temporary dataframe with the 2 classes
as column names. For each student, search through the student grade list for the class names. If a student has grade
entries for both classes, add them to the temporary dataframe. Once all students have been iterated through, calculate
the Spearman rank-order correlation coefficient.
'''
def fill():
    counter = 0
    final = pd.DataFrame(columns=['class_1', 'class_2', 'rho', 'pval', 'n'])
    for (class_1, class1_row_series) in COURSE_COMBINATIONS.iterrows():
        took_both_count = 0
        class_2 = COURSE_COMBINATIONS['class 2'].values[class_1]
        class_1 = COURSE_COMBINATIONS['class 1'].values[class_1]
        df = pd.DataFrame(columns=[class_1, class_2])  # temp dataframe with class1 and class2 as column headers
        for (student_id, student_row_series) in STUDENT_GRADE_LIST.iterrows():
            grade_class_1 = STUDENT_GRADE_LIST[class_1].values[student_id]
            grade_class_2 = STUDENT_GRADE_LIST[class_2].values[student_id]
            if isinstance(grade_class_1, str) and isinstance(grade_class_2, str):
                took_both_count += 1
                tempDf = pd.DataFrame(
                    data={class_1: [grade_class_1.split(',')[1]], class_2: [grade_class_2.split(',')[1]]})
                df = df.append(tempDf, ignore_index=True)
        rho, pval = spearmanr(df[class_1].values, df[class_2].values)
        tempDf = pd.DataFrame(data={'class_1': [class_1], 'class_2': [class_2], 'rho': [rho], 'pval': [pval], 'n' : [took_both_count]})
        final = final.append(tempDf, ignore_index=True)

        print(counter)
        if counter % 1000 == 0:
            final.to_csv('results\\final.csv', index=False)
        counter += 1
    final.to_csv('results\\final.csv', index=False)


if __name__ == "__main__":
    fill()
