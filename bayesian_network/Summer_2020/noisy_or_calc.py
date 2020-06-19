"""
__Author__: Nick Tiede

__Purpose__:
"""
import pandas as pd
from Summer_2020.cartesian_table_creator import create_cartesian_table
from timeit import default_timer as timer


# Takes in a pandas dataframe of course data assuming that the target course is in the last column
# and returns a dataframe of a conditional probability table using noisy-OR
def get_probabilities(dataframe, num_grades):
    num_prereqs = len(dataframe.columns) - 1
    grade_count_tables = []

    # This step might take some time if num_grades is large, scales as 2^(num_grades * 2)
    start_binary_table = timer()
    df_binary_table = create_binary_table(num_grades)
    end_binary_table = timer()
    print('Create binary table time: ' + str(end_binary_table - start_binary_table) + ' sec \n')

    start_count_tables = timer()
    for i in range(0, num_prereqs):
        grade_count_tables.append(create_count_table(dataframe.iloc[:, [i, -1]], df_binary_table))
    end_count_tables = timer()
    print('Create count tables total time: ' + str(end_binary_table - start_binary_table) + ' sec \n')

    return
    # Make list of probability dataframes

    # Make new dataframe with only probability columns


# Creates a truth table structures as the following example based on the number of grade options:
# Prereq    Target
# F D C B A F D C B A
# 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 1
# ...
# 1 1 1 1 1 1 1 1 1 1
# Note that most of these grade options are not possible, as the same course grade can't be both an A and an F
# However, this structure is required when making conditional probability tables in pomegranate
def create_binary_table(num_grades):
    return create_cartesian_table(2, num_grades * 2)


# Returns a DataFrame of a truth table of grades as explained in create_binary_table along with a column
# of counts of each event as found in the dataframe input data
def create_count_table(dataframe, df_structure):
    df_grade_converted = df_structure[0:0]
    num_grades = int(len(df_grade_converted.columns)/2)

    # Converts each non empty grade data to a format that can be compared and counted against df_structure
    for i in range(0, len(dataframe.index)):
        # Creates a template list of all false data
        converted_data_row = [0] * len(df_grade_converted.columns)

        if dataframe.iloc[i, 0] != 'nan' and dataframe.iloc[i, 1] != 'nan':
            converted_data_row[int(dataframe.iloc[i, 0])] = 1
            converted_data_row[int(dataframe.iloc[i, 1]) + num_grades] = 1
            df_grade_converted.loc[len(df_grade_converted)] = converted_data_row

    # Condenses the rows of the converted grade dataframe so duplicates are only listed once and a new
    # column is added for the counts of each instance of data
    df_grade_converted = df_grade_converted.groupby(df_grade_converted.columns.tolist()).size().reset_index(name='count')

    # Gives identical header names to both DataFrames to make merging easier
    headers = list(map(str, range(0, len(df_structure.columns))))
    df_structure.columns = headers
    headers_with_count = headers + ['count']
    df_grade_converted.columns = headers_with_count

    # Makes all values in both DataFrames strings to prevent merge issues
    df_structure = df_structure.astype(str)
    df_grade_converted = df_grade_converted.astype(str)

    # Merges the grade count data into the full truth table
    start = timer()
    df_counts = df_structure.merge(df_grade_converted, on=headers, how='left')
    end = timer()
    print('Merge grade counts table with truth table time: ' + str(end - start) + ' sec \n')

    # Converts NaN values in the counts to their appropriate value of 0
    df_counts['count'] = df_counts['count'].fillna('0')
    print(df_counts)

    return df_counts
