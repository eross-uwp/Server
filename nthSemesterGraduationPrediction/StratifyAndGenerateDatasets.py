import pandas as pd
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 313131
NUMBER_FOLDS = 5
TESTING_TRAINING_DATA_FOLDER = 'data\\test_train\\'
RAW_DATA_FIRST = 'data\\first_term.csv'
RAW_DATA_SECOND = 'data\\second_term.csv'
RAW_DATA_THIRD = 'data\\third_term.csv'
RAW_DATA_ARRAY = [RAW_DATA_FIRST, RAW_DATA_SECOND, RAW_DATA_THIRD]
FIRST_HEADERS = ['first term gpa', 'first term standing']
SECOND_HEADERS = ['first term gpa', 'first term standing', 'second term gpa', 'second term standing']
THIRD_HEADERS = ['first term gpa', 'first term standing', 'second term gpa', 'second term standing', 'third term gpa', 'third term standing']
HEADERS_ARRAY = [FIRST_HEADERS, SECOND_HEADERS, THIRD_HEADERS]
FILENAME_ARRAY = ['first_term_', 'second_term_', 'third_term_']
GRADUATED_HEADER = 'graduated'
TRAIN_PREFIX = 'train_'
TEST_PREFIX = 'test_'


def stratify_fold():
    for i in range(0, 3):
        first_term = pd.read_csv(RAW_DATA_ARRAY[i])
        X = first_term[(HEADERS_ARRAY[i])].copy()
        y = first_term[[GRADUATED_HEADER]].copy()

        skf = StratifiedKFold(n_splits=NUMBER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        loop_count = 0
        X_train = pd.DataFrame(columns=X.columns)
        X_test = pd.DataFrame(columns=X.columns)
        y_train = pd.DataFrame(columns=y.columns)
        y_test = pd.DataFrame(columns=y.columns)
        for train_index, test_index in skf.split(X.values, y.values):
            X_train = X_train.append(X.loc[train_index, :], ignore_index=True)
            X_test = X_test.append(X.loc[test_index, :], ignore_index=True)

            y_train = y_train.append(y.loc[train_index, :], ignore_index=True)
            y_test = y_test.append(y.loc[test_index, :], ignore_index=True)

            (pd.concat(
                [pd.DataFrame(X_train, columns=HEADERS_ARRAY[i]),
                 pd.DataFrame(y_train, columns=[GRADUATED_HEADER])],
                axis=1)).to_csv(TESTING_TRAINING_DATA_FOLDER + FILENAME_ARRAY[i] +
                                TRAIN_PREFIX + str(loop_count + 1) + '.csv', encoding='utf-8', index=False)

            (pd.concat(
                [pd.DataFrame(X_test, columns=HEADERS_ARRAY[i]),
                 pd.DataFrame(y_test, columns=[GRADUATED_HEADER])],
                axis=1)).to_csv(TESTING_TRAINING_DATA_FOLDER + FILENAME_ARRAY[i] +
                                TEST_PREFIX + str(loop_count + 1) + '.csv', encoding='utf-8', index=False)

            loop_count += 1


if __name__ == "__main__":
    stratify_fold()
