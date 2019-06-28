import pandas as pd

if __name__ == '__main__':
    raw_data = pd.read_csv('..\\data\\Curriculum Structure.csv')
    class_list = list(raw_data.postreq.unique())  # all classes in postreq

