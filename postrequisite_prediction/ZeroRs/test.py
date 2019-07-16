import numpy as np
np.random.seed(0)
import pandas as pd
import random
import random
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score
import statistics

possible_grades = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

for key, value in grades_distribution.items():
    grades_distribution[key] = random.random(6)