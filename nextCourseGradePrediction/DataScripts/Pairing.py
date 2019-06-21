import pandas as pd


def fill_class_a(list, a):
    for i, row in classA.iterrows():
        if student == a.at[i, 'student_id']:
            list.append(a.at[i, 'grade'])

    while len(list) < 4:
        list.append('x')


def fill_class_b(list, b):
    student = list[0]

    for i, row in b.iterrows():
        if student == b.at[i, 'student_id']:
            list.append(b.at[i, 'grade'])
        break
    if len(list) < 5:
        list.append('x')


if __name__ == "__main__":
    classA = pd.read_csv('..\\data\\Generated_Pandas\\classA.csv')
    classB = pd.read_csv('..\\data\\Generated_Pandas\\classB.csv')

    idList = classA.student_id.unique()

    ab = pd.DataFrame()

    for student in idList:
        list = [student]
        list.append(student)

        fill_class_a(list, classA)
        fill_class_b(list, classB)

        ab = ab.append({'student_id':list[0], 'classA1': list[1], 'classA2': list[2], 'classA3': list[3], 'classB': list[4]}, ignore_index=True)

    print(ab)
