import numpy
numpy.random.seed(0)
import pandas as pd
from random import choices
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score

GRAPH_FILE_PREFIX = 'graph_term_'
STRATIFIED_DATA_PATH = 'data\\ImmediatePrereqFolds\\'
UNIQUE_COURSE = 'data\\uniqueCourses.csv'
random.seed = 313131
population = [0, 1]
possible_grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F']

def get_training_testing(course_name, number_of_fold):
    return pd.read_csv(STRATIFIED_DATA_PATH + course_name + '_train_' + str(number_of_fold) + '.csv'),\
           pd.read_csv(STRATIFIED_DATA_PATH + course_name + '_test_' + str(number_of_fold) + '.csv')

if __name__ == "__main__":

    unique_course = pd.read_csv(UNIQUE_COURSE)

    for each_course in unique_course['unique_courses']:
        for number_for_fold in (1, 6):
            try:
                train, test = get_training_testing(each_course, number_for_fold)
                y_train = train[each_course].values
                grades_distribution = dict.fromkeys(possible_grades)
                
