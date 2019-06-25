# Using ZeroR to predict if a student will graduate or not

"""
___authors___: Zhiwei Yang
"""

import pandas as pd
import numpy as np
import StratifyAndGenerateDatasets as sd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score


from random import choices
import random


GRAPH_FILE_PREFIX = 'graph_term_'
STRATIFIED_DATA_PATH = 'data\\test_train\\'
random.seed = 313131
population = [0, 1]


#  Iterate through all possible training/testing files and store them in appropriate dataframe.
def get_training_testing(term, number):
    return pd.read_csv(STRATIFIED_DATA_PATH + term + '_term_train_' + str(number) + '.csv'),\
           pd.read_csv(STRATIFIED_DATA_PATH + term + '_term_train_' + str(number) + '.csv')


def zr_predict():
    np.random.seed(sd.RANDOM_SEED)

    for term in ['first', 'second', 'third']:
        prediction_array = np.zeros(0)
        target = np.ones(0)
        for set in range(1, 6):         # looping through each fold
            train, test = get_training_testing(term, set)
            nonz = np.count_nonzero(train['graduated'].values)      # Graduated Count
            train_size = train['graduated'].values.size
            weights = [1-(nonz/train_size), nonz/train_size]        # Weights for predicting

            for eachStudent in range(0, test['graduated'].size):
                prediction_array = np.concatenate((prediction_array, choices(population, weights)), axis=0)
                                                                    # Add prediction
            target = np.concatenate((target, test['graduated']), axis=0)

        tn, fp, fn, tp =confusion_matrix(target, prediction_array).ravel()      # Decompose confusion matrix
        print()
        print(str(term) + ' term result:')
        print('true negative: ', tn, '\nfalse positive: ', fp, '\nfalse negative: ', fn, '\ntrue positive: ', tp)
        print('Precision: ', precision_score(target, prediction_array))
        print('Recall: ', recall_score(target, prediction_array))
        print('ROC_AUC score:' + str(roc_auc_score(target, prediction_array)))


if __name__ == "__main__":
    zr_predict()
