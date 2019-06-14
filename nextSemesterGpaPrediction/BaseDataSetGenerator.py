import random
import pandas as pd
from nextSemesterGpaPrediction.ZeroRModel import predict

# CONSTANTS
RAW_DATA_FILE = 'data\\termGPA.csv'
FINAL_DATA_FILE = 'data\\finalDataSet.csv'
FIRST_COLUMN = 'id'
SECOND_COLUMN = 'prev term number'
THIRD_COLUMN = 'current term number'
FOURTH_COLUMN = 'prev GPA'
FIFTH_COLUMN = 'current GPA'
FINAL_DATA_FRAME_HEADERS = [FIRST_COLUMN, SECOND_COLUMN, THIRD_COLUMN, FOURTH_COLUMN, FIFTH_COLUMN]


def get_term_pairs(raw_data):
    headers = list(raw_data)

    data_frame_for_pairs = pd.DataFrame()

    for student_id in headers:  # for every student id in the raw dataset
        terms = []
        for termNumber in raw_data[student_id].index:  # for every term row of an id column
            if raw_data[student_id].loc[termNumber] > 0:
                terms.append(termNumber)  # copy all term numbers that relate to a non-zero gpa to a temp array
        if len(terms) > 1:  # as long as a student id is related to at least 2 terms of non-zero gpa
            random_term_index = random.randrange(1, len(terms))
            second_term = terms[random_term_index]
            first_term = terms[random_term_index - 1]
            data_frame_for_pairs[student_id] = [first_term, second_term]  # random term and the term previous to it for each id
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


if __name__ == "__main__":
    rawData = pd.read_csv(RAW_DATA_FILE, index_col="index")  # our raw dataset
    termPairsDataFrame = get_term_pairs(rawData) # Get a random pair of terms for each applicable student id
    finalDataFrame = generate_final_dataset(termPairsDataFrame, rawData) # Get the corresponding gpa for each term pair
    finalDataFrame.to_csv(FINAL_DATA_FILE, encoding='utf-8', index=False)
    print(predict(finalDataFrame[FOURTH_COLUMN])) # Run the ZeroRModel predict function
