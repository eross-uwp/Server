# Using ZeroR to predict if a student will graduate or not

"""
___authors___: Zhiwei Yang
"""

import pandas as pd
import numpy as np
import StratifyAndGenerateDatasets as sd
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, accuracy_score

GRAPH_FILE_PREFIX = 'graph_term_'
STRATIFIED_DATA_PATH = 'data\\test_train\\'


#  Iterate through all possible training/testing files and store them in appropriate arrays.
def get_training_testing(term, number):
    return pd.read_csv(STRATIFIED_DATA_PATH + term + '_term_train_' + str(number) + '.csv'),\
           pd.read_csv(STRATIFIED_DATA_PATH + term + '_term_train_' + str(number) + '.csv') # looping through each fold


def zr_predict():
    np.random.seed(sd.RANDOM_SEED)
    for term in ['first', 'second', 'third']:
        prediction_array = np.zeros(0)
        target = np.ones(0)
        for set in range(1, 6):
            train, test = get_training_testing(term, set)
            nonz = np.count_nonzero(train['graduated'].values)          # Graduated Count
            train_size = train['graduated'].values.size

            if nonz/train_size < 0.5:
                prediction_array = np.concatenate((prediction_array, np.zeros(test['graduated'].size)), axis=0)
                target = np.concatenate((target, test['graduated']), axis=0)    # Add prediction
            else:
                prediction_array = np.concatenate((prediction_array, np.ones(test['graduated'].size)), axis=0)
                target = np.concatenate((target, test['graduated']), axis=0)    # Add prediction

        tn, fp, fn, tp =confusion_matrix(target, prediction_array).ravel()      # Decompose confusion matrix
        print()
        print(str(term) + ' term result:')
        print('true negative: ', tn, '\nfalse positive: ', fp, '\nfalse negative: ', fn, '\ntrue positive: ', tp)
        print('Precision score: ', precision_score(target, prediction_array))
        print('Recall score: ', recall_score(target, prediction_array))
        print('ROC_AUC score:' + str(roc_auc_score(target, prediction_array)))
        print('Accuracy score:' + str(accuracy_score(target, prediction_array)))


if __name__ == "__main__":
    zr_predict()