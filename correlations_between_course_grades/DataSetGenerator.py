# create pairs of all classes first.

import itertools

import pandas as pd

COURSE_LIST = pd.read_csv('..\\Data\\unique_courses.csv')['courses']
COURSE_COMBINATIONS_FILEPATH = '..\\Data\\course_combinations.csv'


def get_class_pairs():
    """
    Generate all possible combinations of courses given our dataset.
    :return:
    """
    all_pairs = pd.DataFrame(list(itertools.combinations(COURSE_LIST.values, 2)), columns=['class 1', 'class 2'])
    all_pairs.to_csv(COURSE_COMBINATIONS_FILEPATH, encoding='utf-8', index=False)


if __name__ == "__main__":
    get_class_pairs()
