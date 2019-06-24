# Using ZeroR to predict if a student will graduate or not

"""
___authors___: Zhiwei Yang, Austin FitzGerald
"""

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import StratifyAndGenerateDatasets as sd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


RESULTS_FOLDER = 'ZeroR2Results\\'
GRAPH_FILE_PREFIX = 'graph_term_'
RESULTS_TEXTFILE = 'ZeroR2_Results.txt'

prediction = -1
predictionArray = np.zeros(1)

#  Iterate through all possible training/testing files and store them in appropriate arrays.
def get_training_testing():
    return pd.read_csv('C:\\Users\\yangz\\Documents\\GitHub\\Server\\nthSemesterGraduationPrediction\\data\\test_train\\first_term_train_1.csv'), pd.read_csv('C:\\Users\\yangz\\Documents\\GitHub\\Server\\nthSemesterGraduationPrediction\\data\\test_train\\first_term_test_1.csv')

def zr_predict():
    np.random.seed(sd.RANDOM_SEED)
    train, test = get_training_testing()
    nonz = np.count_nonzero(train['graduated'].values)
    print(nonz)
    train_size = train['graduated'].values.size
    print(train_size)
    if (nonz/train_size < 0.5):
        prediction = 0
    else:
        prediction = 1

    if prediction == 0:
        predictionArray = np.zeros(test['graduated'].size)
        print(predictionArray)
    if prediction == 1:
        predictionArray = np.ones(test['graduated'].size)
        print(predictionArray)
    else: print('Not trained yet')
    print(test['graduated'].values)
    tn, fp, fn, tp =confusion_matrix(test['graduated'].values, predictionArray).ravel()
    print('true negative: ', tn, '\nfalse positive: ', fp, '\nfalse negative: ', fn, '\ntrue positive: ', tp)

    print('Precision: ', precision_score(test['graduated'].values, predictionArray))
    print('Recall: ', recall_score(test['graduated'].values, predictionArray))
    print('F1 score: ', f1_score(test['graduated'].values, predictionArray))

if __name__ == "__main__":
    zr_predict()
