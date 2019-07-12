# Generates a dataset with matching term/gpa

"""
___authors___: Zhiwei Yang and Austin FitzGerald
"""

import random
import pandas as pd
from ZeroRModel import predict
from sklearn.model_selection import StratifiedKFold

# CONSTANTS
RAW_DATA_FILE = 'data\\term_gpa_only_springfall.csv'
FINAL_DATA_FILE = 'data\\final_student_random_gpa_pair.csv'
TESTING_TRAINING_DATA_FOLDER = 'data\\test_train\\'
TRAIN_PREFIX = 'train_'
TEST_PREFIX = 'test_'
FIRST_COLUMN = 'id'
SECOND_COLUMN = 'prev term number'
THIRD_COLUMN = 'next term number'
FOURTH_COLUMN = 'prev GPA'
FIFTH_COLUMN = 'next GPA'
FINAL_DATA_FRAME_HEADERS = [FIRST_COLUMN, SECOND_COLUMN, THIRD_COLUMN, FOURTH_COLUMN, FIFTH_COLUMN]
RANDOM_SEED = 313131
NUMBER_OF_FOLDS = 5


def get_term_pairs(raw_data):
    random.seed(RANDOM_SEED)  # Make sure we get the same dataset every time
    headers = list(raw_data)

    data_frame_for_pairs = pd.DataFrame()

    for student_id in headers:  # for every student id in the raw dataset
        term_pairs = []
        # get each pair of consecutive terms where the student does not have a null grade
        for first_termnumber, second_termnumber in zip(raw_data[student_id].index, raw_data[student_id].index[1:]):
            if raw_data[student_id].loc[first_termnumber] != '' and raw_data[student_id].loc[second_termnumber] != '':
                term_pairs.append([first_termnumber, second_termnumber])

        if len(term_pairs) > 1:  # as long as a student id is related to at least 2 terms of non-null gpa
            random_term_index = random.randrange(1, len(term_pairs))
            second_term = term_pairs[random_term_index][1]
            first_term = term_pairs[random_term_index][0]
            # first and second terms from random term pair added to dataframe
            data_frame_for_pairs[student_id] = [first_term, second_term]
    return data_frame_for_pairs


def generate_final_dataset(term_pairs_data_frame, raw_data_frame):
    final_data_frame = pd.DataFrame(columns=FINAL_DATA_FRAME_HEADERS)

    ids = list(term_pairs_data_frame)

    # for each student id, copy the 2 terms we found and the corresponding GPAs to the final dataset
    for student_id in ids:
        final_data_frame = final_data_frame.append({FIRST_COLUMN: student_id,
                                                    SECOND_COLUMN: term_pairs_data_frame[student_id][0],
                                                    THIRD_COLUMN: term_pairs_data_frame[student_id][1],
                                                    FOURTH_COLUMN: raw_data_frame[student_id][
                                                        term_pairs_data_frame[student_id][0]],
                                                    FIFTH_COLUMN: raw_data_frame[student_id][
                                                        term_pairs_data_frame[student_id][1]]}, ignore_index=True)
    return final_data_frame


def stratify_and_five_fold(final_data_frame):
    x = final_data_frame[SECOND_COLUMN].values  # get numpy array of prev and next terms
    y = final_data_frame[THIRD_COLUMN].values

    prev_gpa = final_data_frame[FOURTH_COLUMN].values  # get numpy array of prev and next GPAs
    next_gpa = final_data_frame[FIFTH_COLUMN].values

    skf = StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_SEED)  # setup stratified k fold

    loop_count = 0
    for train_index, test_index in skf.split(x, y):
        x_train_term, x_test_term = x[train_index], x[test_index]
        x_train_gpa, x_test_gpa = prev_gpa[train_index], prev_gpa[test_index]
        y_train_term, y_test_term = y[train_index], y[test_index]
        y_train_gpa, y_test_gpa = next_gpa[train_index], next_gpa[test_index]

        # write the new testing and training sets to csv files
        (pd.concat(
            [pd.DataFrame(x_train_term, columns=[SECOND_COLUMN]),
             pd.DataFrame(y_train_term, columns=[THIRD_COLUMN]),
             pd.DataFrame(x_train_gpa, columns=[FOURTH_COLUMN]),
             pd.DataFrame(y_train_gpa, columns=[FIFTH_COLUMN])],
            axis=1)).to_csv(TESTING_TRAINING_DATA_FOLDER +
                            TRAIN_PREFIX + str(loop_count + 1) + '.csv', encoding='utf-8', index=False)
        (pd.concat(
            [pd.DataFrame(x_test_term, columns=[SECOND_COLUMN]),
             pd.DataFrame(y_test_term, columns=[THIRD_COLUMN]),
             pd.DataFrame(x_test_gpa, columns=[FOURTH_COLUMN]),
             pd.DataFrame(y_test_gpa, columns=[FIFTH_COLUMN])],
            axis=1)).to_csv(TESTING_TRAINING_DATA_FOLDER +
                            TEST_PREFIX + str(loop_count + 1) + '.csv', encoding='utf-8', index=False)
        loop_count += 1


if __name__ == "__main__":
    rawData = pd.read_csv(RAW_DATA_FILE, index_col="index").fillna('')  # our raw dataset
    termPairsDataFrame = get_term_pairs(rawData)  # Get a random pair of terms for each applicable student id
    finalDataFrame = generate_final_dataset(termPairsDataFrame, rawData)  # Get the corresponding gpa for each term pair
    finalDataFrame.to_csv(FINAL_DATA_FILE, encoding='utf-8', index=False)

    finalDataFrame = pd.read_csv(FINAL_DATA_FILE, index_col=FIRST_COLUMN)  # yes this is a bit hacky.

    stratify_and_five_fold(finalDataFrame)

    print(predict(finalDataFrame[FOURTH_COLUMN]))  # Run the ZeroRModel predict function
