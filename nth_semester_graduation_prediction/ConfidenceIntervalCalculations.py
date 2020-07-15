import random

import numpy as np
import pandas as pd
from sklearn import metrics

__LOGISTIC_REGRESSION_PREDICTION_FOLDER = 'LogisticRegressionResults\\prediction output\\'
__GRADIENT_BOOSTED_TREES_PREDICTION_FOLDER = 'GBTClassifierResults\\prediction output\\'
__ZERO_R_PREDICTION_FOLDER = 'ZeroR_Results\\most_frequent_prediction_output\\'
__RESULTS_FOLDER = 'ConfidenceIntervalResults\\'

__NUMBER_TERMS = 10
__NUMBER_ITER = 10000
__CONFIDENCE_INTERVAL = 0.95

np.random.seed(313131)


# https://stackoverflow.com/a/43886290
def round_school(x_list):
    temp_list = []
    for x in x_list:
        if x < 0:
            return 0
        else:
            i, f = divmod(x, 1)
            temp_list.append(int(i + ((f >= 0.5) if (x > 0) else (f > 0.5))))
    return temp_list


def get_all():
    massive_actual = []
    massive_logistic_probs = []
    massive_gbt_probs = []
    massive_zeror_probs = []
    for term_num in range(0, __NUMBER_TERMS):
        logistic_df = pd.read_csv(__LOGISTIC_REGRESSION_PREDICTION_FOLDER + 'term_' + str(term_num + 1) + '.csv')
        gbt_df = pd.read_csv(__GRADIENT_BOOSTED_TREES_PREDICTION_FOLDER + 'term_' + str(term_num + 1) + '.csv')
        zeror_df = pd.read_csv(__ZERO_R_PREDICTION_FOLDER + 'term_' + str(term_num + 1) + '.csv')

        massive_actual += list(logistic_df['actual'].values)
        massive_logistic_probs += list(logistic_df['prob of grad'].values)
        massive_gbt_probs += list(gbt_df['prob of grad'].values)
        massive_zeror_probs += list(zeror_df['prob of grad'].values)

    n_samples = len(massive_actual)

    auc_logistic_list = []
    accuracy_logistic_list = []
    auc_gbt_list = []
    accuracy_gbt_list = []
    auc_zeror_list = []
    accuracy_zeror_list = []


    # we want '__NUMBER_ITER' number of auc/accuracy scores per model
    for iter in range(0, __NUMBER_ITER):
        big_actual_scores = []
        big_logistic_probs = []
        big_gbt_probs = []
        big_zeror_probs = []

        # do random with replacement 'n_samples' times
        for sample in range(0, n_samples):
            index = random.randint(0, n_samples - 1)
            big_actual_scores.append(massive_actual[index])
            big_logistic_probs.append(massive_logistic_probs[index])
            big_gbt_probs.append(massive_gbt_probs[index])
            big_zeror_probs.append(massive_zeror_probs[index])

        # get scores for models from big lists
        auc_logistic_list.append(metrics.roc_auc_score(big_actual_scores, big_logistic_probs))
        accuracy_logistic_list.append(metrics.accuracy_score(big_actual_scores, round_school(big_logistic_probs)))
        auc_gbt_list.append(metrics.roc_auc_score(big_actual_scores, big_gbt_probs))
        accuracy_gbt_list.append(metrics.accuracy_score(big_actual_scores, round_school(big_gbt_probs)))
        auc_zeror_list.append(metrics.roc_auc_score(big_actual_scores, big_zeror_probs))
        accuracy_zeror_list.append(metrics.accuracy_score(big_actual_scores, round_school(big_zeror_probs)))

    auc_logistic_list.sort()
    accuracy_logistic_list.sort()
    auc_gbt_list.sort()
    accuracy_gbt_list.sort()
    auc_zeror_list.sort()
    accuracy_zeror_list.sort()

    upper_index = round(__NUMBER_ITER * (1 - ((1 - __CONFIDENCE_INTERVAL) / 2)))
    lower_index = round(__NUMBER_ITER * (0 + ((1 - __CONFIDENCE_INTERVAL) / 2)))

    auc_logistic = [auc_logistic_list[lower_index], auc_logistic_list[upper_index]]
    accuracy_logistic = [accuracy_logistic_list[lower_index], accuracy_logistic_list[upper_index]]
    auc_gbt = [auc_gbt_list[lower_index], auc_gbt_list[upper_index]]
    accuracy_gbt = [accuracy_gbt_list[lower_index], accuracy_gbt_list[upper_index]]
    auc_zeror = [auc_zeror_list[lower_index], auc_zeror_list[upper_index]]
    accuracy_zeror = [accuracy_zeror_list[lower_index], accuracy_zeror_list[upper_index]]

    with open(__RESULTS_FOLDER + 'LogisticRegression_all' + '.txt', "w") as text_file:
        text_file.write(
            'AUC = [' + str(auc_logistic[0]) + ', ' + str(auc_logistic[1]) +
            '], Accuracy = [' + str(accuracy_logistic[0]) + ', ' + str(accuracy_logistic[1]) + ']')

    with open(__RESULTS_FOLDER + 'GBTClassifier_all' + '.txt', "w") as text_file:
        text_file.write(
            'AUC = [' + str(auc_gbt[0]) + ', ' + str(auc_gbt[1]) +
            '], Accuracy = [' + str(accuracy_gbt[0]) + ', ' + str(accuracy_gbt[1]) + ']')

    with open(__RESULTS_FOLDER + 'ZerorRMostFrequent_all' + '.txt', "w") as text_file:
        text_file.write(
            'AUC = [' + str(auc_zeror[0]) + ', ' + str(auc_zeror[1]) +
            '], Accuracy = [' + str(accuracy_zeror[0]) + ', ' + str(accuracy_zeror[1]) + ']')


if __name__ == "__main__":
    get_all()

    for term_num in range(0, __NUMBER_TERMS):
        logistic_df = pd.read_csv(__LOGISTIC_REGRESSION_PREDICTION_FOLDER + 'term_' + str(term_num+1) + '.csv')
        gbt_df = pd.read_csv(__GRADIENT_BOOSTED_TREES_PREDICTION_FOLDER + 'term_' + str(term_num+1) + '.csv')
        zeror_df = pd.read_csv(__ZERO_R_PREDICTION_FOLDER + 'term_' + str(term_num+1) + '.csv')

        actual_scores = logistic_df['actual'].values
        logistic_probs = logistic_df['prob of grad'].values
        gbt_probs = gbt_df['prob of grad'].values
        zeror_probs = zeror_df['prob of grad'].values

        n_samples = len(actual_scores)

        auc_logistic_list = []
        accuracy_logistic_list = []
        auc_gbt_list = []
        accuracy_gbt_list = []
        auc_zeror_list = []
        accuracy_zeror_list = []

        # we want '__NUMBER_ITER' number of auc/accuracy scores per model
        for iter in range(0, __NUMBER_ITER):
            big_actual_scores = []
            big_logistic_probs = []
            big_gbt_probs = []
            big_zeror_probs = []

            # do random with replacement 'n_samples' times
            for sample in range(0, n_samples):
                index = random.randint(0, n_samples-1)
                big_actual_scores.append(actual_scores[index])
                big_logistic_probs.append(logistic_probs[index])
                big_gbt_probs.append(gbt_probs[index])
                big_zeror_probs.append(zeror_probs[index])

            # get scores for models from big lists
            auc_logistic_list.append(metrics.roc_auc_score(big_actual_scores, big_logistic_probs))
            accuracy_logistic_list.append(metrics.accuracy_score(big_actual_scores, round_school(big_logistic_probs)))
            auc_gbt_list.append(metrics.roc_auc_score(big_actual_scores, big_gbt_probs))
            accuracy_gbt_list.append(metrics.accuracy_score(big_actual_scores, round_school(big_gbt_probs)))
            auc_zeror_list.append(metrics.roc_auc_score(big_actual_scores, big_zeror_probs))
            accuracy_zeror_list.append(metrics.accuracy_score(big_actual_scores, round_school(big_zeror_probs)))

        auc_logistic_list.sort()
        accuracy_logistic_list.sort()
        auc_gbt_list.sort()
        accuracy_gbt_list.sort()
        auc_zeror_list.sort()
        accuracy_zeror_list.sort()

        upper_index = round(__NUMBER_ITER * (1 - ((1-__CONFIDENCE_INTERVAL)/2)))
        lower_index = round(__NUMBER_ITER * (0 + ((1-__CONFIDENCE_INTERVAL)/2)))

        auc_logistic = [auc_logistic_list[lower_index], auc_logistic_list[upper_index]]
        accuracy_logistic = [accuracy_logistic_list[lower_index], accuracy_logistic_list[upper_index]]
        auc_gbt = [auc_gbt_list[lower_index], auc_gbt_list[upper_index]]
        accuracy_gbt = [accuracy_gbt_list[lower_index], accuracy_gbt_list[upper_index]]
        auc_zeror = [auc_zeror_list[lower_index], auc_zeror_list[upper_index]]
        accuracy_zeror = [accuracy_zeror_list[lower_index], accuracy_zeror_list[upper_index]]

        with open(__RESULTS_FOLDER + 'LogisticRegression' + str(term_num + 1) + '.txt', "w") as text_file:
            text_file.write(
                'AUC = [' + str(auc_logistic[0]) + ', ' + str(auc_logistic[1]) +
                '], Accuracy = [' + str(accuracy_logistic[0]) + ', ' + str(accuracy_logistic[1]) + ']')

        with open(__RESULTS_FOLDER + 'GBTClassifier' + str(term_num + 1) + '.txt', "w") as text_file:
            text_file.write(
                'AUC = [' + str(auc_gbt[0]) + ', ' + str(auc_gbt[1]) +
                '], Accuracy = [' + str(accuracy_gbt[0]) + ', ' + str(accuracy_gbt[1]) + ']')

        with open(__RESULTS_FOLDER + 'ZerorRMostFrequent' + str(term_num + 1) + '.txt', "w") as text_file:
            text_file.write(
                'AUC = [' + str(auc_zeror[0]) + ', ' + str(auc_zeror[1]) +
                '], Accuracy = [' + str(accuracy_zeror[0]) + ', ' + str(accuracy_zeror[1]) + ']')