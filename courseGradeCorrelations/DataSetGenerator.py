# create pairs of all classes first. then find students who took both and record grade for each course.

import pandas as pd
import numpy as np
import itertools


def get_class_pairs():
    raw_dataset = pd.read_csv('data\\raw_courses.csv')['courses']
    all_pairs = pd.DataFrame(list(itertools.combinations(raw_dataset.values, 2)), columns=['class 1', 'class 2'])
    all_pairs.to_csv('data\\course_combinations.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    # get_class_pairs()
    print('done')
