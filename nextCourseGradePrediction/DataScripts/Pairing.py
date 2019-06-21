import pandas as pd

if __name__ == "__main__":
    calc1 = pd.read_csv('..\\data\\Generated_Pandas\\class1.csv')
    calc2 = pd.read_csv('..\\data\\Generated_Pandas\\class2.csv')
    pair = pd.DataFrame(columns=['student_id', 'class1', 'class2'])

    for i, row in calc1.iterrows():
        studentId = calc1.at[i, 'student_id']
        for j, row2 in calc2.iterrows():
            if studentId == calc2.at[j, 'student_id']:
                pair.at[i, 'student_id'] = studentId
                pair.at[i, 'class1'] = calc1.at[i, 'grade']
                pair.at[i, 'class2'] = calc2.at[j, 'grade']

    pair.to_csv('..\\data\\Generated_Pandas\\classPairs.csv')
