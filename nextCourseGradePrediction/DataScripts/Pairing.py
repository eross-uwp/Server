import pandas as pd

if __name__ == "__main__":
    calc1 = pd.read_csv('..\\data\\Generated_Pandas\\classA.csv')
    calc2 = pd.read_csv('..\\data\\Generated_Pandas\\classB.csv')
    pair = pd.DataFrame(columns=['student_id', 'classA1', 'ClassA2', 'ClassA3', 'classB'])


