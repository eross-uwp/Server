"""
__Author__: Nick Tiede

__Purpose__: To create a pomegranate conditional probability table for the target course
"""
import pandas as pd
from itertools import product
from pomegranate import ConditionalProbabilityTable
from Summer_2020.cartesian_table_creator import create_cartesian_table


# Takes in the number of prereqs and the number of grades
# Creates a list of rows with every possible event and an equal probability for each
# This should be used when learning from samples with pomegranate
def create_con_prob_table(num_prereqs, num_grades, states):

    # Creates the cartesian product of the grades as a DataFrame
    df_events = create_cartesian_table(num_grades, num_prereqs + 1)

    # Adds a column of probabilities as floats to the DataFrame
    df_events[len(df_events.columns)] = 1/num_grades

    return ConditionalProbabilityTable(df_events.values.tolist(), get_disc_dist_list(states))


# Returns a list of discrete distributions from a list of states
def get_disc_dist_list(states):
    return [state.distribution for state in states]


# Creates a standard Bayesian network cpt manually
def create_cpt(df_data, num_grades, num_prereqs):
    df_grades = df_data.copy().astype(str)

    # Sets nan values to the most common grade
    for j in range(0, len(df_data.columns)):
        mode_grade = df_data.iloc[:, j].mode()
        for i in range(0, len(df_data.index)):
            if df_data.iloc[i, j] == 'nan':
                df_data.iat[i, j] = mode_grade

    df_structure = create_cartesian_table(num_grades, num_prereqs+1)

    # Condenses the rows of the filtered grade dataframe so duplicates are only listed once and a new
    # column is added for the counts of each instance of data
    df_grades = df_grades.groupby(df_grades.columns.tolist()).size().reset_index(name='count')

    # Gives identical header names to both DataFrames to make merging easier
    headers = list(map(str, range(0, len(df_structure.columns))))
    df_structure.columns = headers
    df_grades.columns = headers + ['count']

    # Makes all values in both DataFrames strings to prevent merge issues
    df_structure = df_structure.astype(str)
    df_grades = df_grades.astype(str)

    # Merges the grade count data into the full truth table
    df_counts = df_structure.merge(df_grades, on=headers, how='left')

    # Converts NaN values in the counts to their appropriate value of 0
    df_counts['count'] = df_counts['count'].fillna('0')

    df_counts['count'] = df_counts['count'].astype(int)

    # Calculates the probabilities
    for i in range(num_grades ** num_prereqs):
        row_i_min = int(i*num_grades)
        row_i_max = int((i*num_grades) + num_grades)
        count_sum = df_counts.iloc[row_i_min:row_i_max, -1].sum()

        if int(count_sum) == 0:
            df_counts.iloc[row_i_min:row_i_max, -1] = 1/num_grades
            # Fixes the random predict problem by adding a small amount of probability to the most common grade
            # While also keeping the sums added to 1
            df_counts.iloc[row_i_min:row_i_max, -1] -= 0.00001
            mode_grade = int(df_data.iloc[:, -2].mode())
            df_counts.iat[row_i_min + mode_grade, -1] += 0.00001 * num_grades
        else:
            prob_modifier = 1/count_sum
            df_counts.iloc[row_i_min:row_i_max, -1] *= prob_modifier

    df_counts.rename(columns={'count': 'probability'}, inplace=True)

    return df_counts
