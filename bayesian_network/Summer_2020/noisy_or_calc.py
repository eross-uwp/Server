"""
__Author__: Nick Tiede

__Purpose__: To use Noisy-OR Bayesian Network methods to create conditional probability table DataFrames based
             on given data to reduce required sample size. The results of this file replace the need to use
             the fitting functions in pomegranate. These tables are meant to be converted to pomegranate
             ConditionalProbabilityTables along with given data DiscreteDistributions to create the Bayesian
             Network model structure for prediction using pomegranate predict methods.
"""
import pandas as pd
from Summer_2020.cartesian_table_creator import create_cartesian_table
from timeit import default_timer as timer


# Takes in a pandas dataframe of course data assuming that the target course is in the last column
# and returns a dataframe of a conditional probability table using noisy-OR
def get_probabilities(dataframe, num_grades):
    num_prereqs = len(dataframe.columns) - 1

    # This step might take some time if num_grades is large, scales as 2^(num_grades * 2)
    start_binary_table = timer()
    df_binary_table = create_binary_table(num_grades)
    end_binary_table = timer()
    print('Create binary table time: ' + str(end_binary_table - start_binary_table) + ' sec \n')

    # Creates a list of each prereq to target course grade count table
    start_count_tables = timer()
    grade_count_tables = []
    for i in range(0, num_prereqs):
        grade_count_tables.append(create_count_table(dataframe.iloc[:, [i, -1]], df_binary_table))
    end_count_tables = timer()
    print('Create count tables total time: ' + str(end_binary_table - start_binary_table) + ' sec \n')

    # Turns the count tables into conditional probability table DataFrames
    probability_tables = []
    for item in grade_count_tables:
        probability_tables.append(create_probability_table(item, num_grades))

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
# The non possible options are then removed from the DataFrame
# This may not be the most efficient way to do this, but it is easy
def create_binary_table(num_grades):
    df_cartesian = create_cartesian_table(2, num_grades * 2).astype(int)
    df_possible = df_cartesian.loc[(df_cartesian.iloc[:, 0: num_grades].sum(axis=1) == 1), ]
    df_possible = df_possible.loc[(df_cartesian.iloc[:, num_grades: num_grades*2].sum(axis=1) == 1), ]
    return df_possible


# Returns a DataFrame of a truth table of grades as explained in create_binary_table along with a column
# of counts of each event as found in the dataframe input data
def create_count_table(dataframe, df_structure):
    df_grade_converted = df_structure[0:0]
    num_grades = int(len(df_grade_converted.columns)/2)

    # Converts each non empty grade data to a format that can be compared and counted against df_structure
    for i in range(0, len(dataframe.index)):
        # Creates a template list of all false data
        converted_data_row = [0] * len(df_grade_converted.columns)

        # Filters out nan data
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
    df_grade_converted.columns = headers + ['count']

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


# Turns a grade count table into a conditional probability table for each prereq to target, returns a DataFrame
def create_probability_table(df_count_table, num_grades):
    df_prob_table = df_count_table
    df_prob_table[['count']] = df_prob_table[['count']].astype(float)

    for prereq_grade in range(0, num_grades):
        count_sum = df_prob_table[df_prob_table[str(prereq_grade)] == '1']['count'].sum()
        if count_sum != 0:
            df_prob_table.loc[df_prob_table[str(prereq_grade)] == '1', 'count'] *= 1/count_sum

    df_prob_table.rename(columns={'count': 'probability'}, inplace=True)
    return df_prob_table