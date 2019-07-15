import random

import pandas as pd
from sklearn import metrics

__LOGISTIC_REGRESSION_PREDICTION_FOLDER = 'LogisticRegressionResults\\prediction output\\'
__GRADIENT_BOOSTED_TREES_PREDICTION_FOLDER = 'GBTClassifierResults\\prediction output\\'
__ZERO_R_PREDICTION_FOLDER = ''
__RESULTS_FOLDER = 'ConfidenceIntervalResults\\'

__NUMBER_TERMS = 10
__NUMBER_ITER = 10000
__CONFIDENCE_INTERVAL = 0.95

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


if __name__ == "__main__":
    for term_num in range(0, __NUMBER_TERMS):
        logistic_df = pd.read_csv(__LOGISTIC_REGRESSION_PREDICTION_FOLDER + 'term_' + str(term_num+1) + '.csv')
        gbt_df = pd.read_csv(__GRADIENT_BOOSTED_TREES_PREDICTION_FOLDER + 'term_' + str(term_num+1) + '.csv')

        actual_scores = logistic_df['actual'].values
        logistic_probs = logistic_df['prob of grad'].values
        gbt_probs = gbt_df['prob of grad'].values

        n_samples = len(actual_scores)

        auc_logistic_list = []
        accuracy_logistic_list = []
        auc_gbt_list = []
        accuracy_gbt_list = []

        # we want '__NUMBER_ITER' number of auc/accuracy scores per model
        for iter in range(0, __NUMBER_ITER):
            big_actual_scores = []
            big_logistic_probs = []
            big_gbt_probs = []

            # do random with replacement 'n_samples' times
            for sample in range(0, n_samples):
                index = random.randint(0, n_samples-1)
                big_actual_scores.append(actual_scores[index])
                big_logistic_probs.append(logistic_probs[index])
                big_gbt_probs.append(gbt_probs[index])

            # get scores for models from big lists
            auc_logistic_list.append(metrics.roc_auc_score(big_actual_scores, big_logistic_probs))
            accuracy_logistic_list.append(metrics.accuracy_score(big_actual_scores, round_school(big_logistic_probs)))
            auc_gbt_list.append(metrics.roc_auc_score(big_actual_scores, big_gbt_probs))
            accuracy_gbt_list.append(metrics.accuracy_score(big_actual_scores, round_school(big_gbt_probs)))

        auc_logistic_list.sort()
        accuracy_logistic_list.sort()
        auc_gbt_list.sort()
        accuracy_gbt_list.sort()

        upper_index = round(__NUMBER_ITER * (1 - ((1-__CONFIDENCE_INTERVAL)/2)))
        lower_index = round(__NUMBER_ITER * (0 + ((1-__CONFIDENCE_INTERVAL)/2)))

        auc_logistic = [auc_logistic_list[lower_index], auc_logistic_list[upper_index]]
        accuracy_logistic = [accuracy_logistic_list[lower_index], accuracy_logistic_list[upper_index]]
        auc_gbt = [auc_gbt_list[lower_index], auc_gbt_list[upper_index]]
        accuracy_gbt = [accuracy_gbt_list[lower_index], accuracy_gbt_list[upper_index]]

        with open(__RESULTS_FOLDER + 'LogisticRegression' + str(term_num + 1) + '.txt', "w") as text_file:
            text_file.write(
                'AUC = [' + str(auc_logistic[0]) + ', ' + str(auc_logistic[1]) +
                ', Accuracy = [' + str(accuracy_logistic[0]) + ', ' + str(accuracy_logistic[1]))

        with open(__RESULTS_FOLDER + 'GBTClassifier' + str(term_num + 1) + '.txt', "w") as text_file:
            text_file.write(
                'AUC = [' + str(auc_gbt[0]) + '], ' + str(auc_gbt[1]) +
                ', Accuracy = [' + str(accuracy_gbt[0]) + ', ' + str(accuracy_gbt[1]) + ']')