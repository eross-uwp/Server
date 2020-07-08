"""
__Author__: Nick Tiede

__Purpose__: To use our Noisy-Avg Bayesian Network methods to create conditional probability table DataFrames based
             on given data to reduce required sample size. The results of this file replace the need to use
             the fitting functions in pomegranate. These tables are meant to be converted to pomegranate
             ConditionalProbabilityTables along with given data DiscreteDistributions to create the Bayesian
             Network model structure for prediction using pomegranate predict methods.
"""
import pandas as pd
from Summer_2020.cartesian_table_creator import create_cartesian_table
from timeit import default_timer as timer


# Takes in a pandas dataframe of course data assuming that the target course is in the last column
# and returns a dataframe of a conditional probability table using our noisy-avg
def create_target_cpt(dataframe, num_grades):
    start_time = timer()  # Gives total time in this function

    num_prereqs = len(dataframe.columns) - 1

    # Creates the auxiliary node conditional probability table structure without the probability column
    aux_cpt_structure = create_cartesian_table(num_grades, 2)

    # Creates the target node conditional probability table structure without the probability column
    final_cpt_structure = create_cartesian_table(num_grades, num_prereqs+1)

    # Creates a DataFrame with the averages of each combination of auxiliary grades
    df_averages = find_averages(final_cpt_structure)

    # Creates a list of each prereq to target course grade count table
    grade_count_tables = []
    for i in range(0, num_prereqs):
        grade_count_tables.append(create_count_table(dataframe.iloc[:, [i, -1]], aux_cpt_structure, num_grades))

    # Turns the count tables into conditional probability table DataFrames
    probability_tables = []
    for item in grade_count_tables:
        probability_tables.append(create_probability_table(item, num_grades))

    end_time = timer()
    print('Create conditional probabilities total time: ' + str(end_time - start_time) + ' sec \n')
    return


# Returns a DataFrame with counts of prereq to target course grade instances in a CPT like structure
def create_count_table(dataframe, df_structure, num_grades):
    df_grades_filtered = df_structure[0:0]

    # Puts each non empty grade data into a new DataFrame that can be counted
    for i in range(0, len(dataframe.index)):
        # Creates a template row
        filtered_data_row = [0, 0]

        # Filters out nan data
        if dataframe.iloc[i, 0] != 'nan' and dataframe.iloc[i, 1] != 'nan':
            filtered_data_row[0] = int(dataframe.iloc[i, 0])
            filtered_data_row[1] = int(dataframe.iloc[i, 1])
            df_grades_filtered.loc[len(df_grades_filtered)] = filtered_data_row

    # Condenses the rows of the filtered grade dataframe so duplicates are only listed once and a new
    # column is added for the counts of each instance of data
    df_grades_filtered = df_grades_filtered.groupby(df_grades_filtered.columns.tolist()).size().reset_index(name='count')

    # Gives identical header names to both DataFrames to make merging easier
    headers = list(map(str, range(0, len(df_structure.columns))))
    df_structure.columns = headers
    df_grades_filtered.columns = headers + ['count']

    # Makes all values in both DataFrames strings to prevent merge issues
    df_structure = df_structure.astype(str)
    df_grades_filtered = df_grades_filtered.astype(str)

    # Merges the grade count data into the full truth table
    df_counts = df_structure.merge(df_grades_filtered, on=headers, how='left')

    # Converts NaN values in the counts to their appropriate value of 0
    df_counts['count'] = df_counts['count'].fillna('0')

    return df_counts


# Turns a grade count table into a conditional probability table for each prereq to target, returns a DataFrame
def create_probability_table(df_count_table, num_grades):
    df_prob_table = df_count_table
    df_prob_table[['count']] = df_prob_table[['count']].astype(float)

    for prereq_grade in range(0, num_grades):
        count_sum = df_prob_table[df_prob_table['0'] == str(prereq_grade)]['count'].sum()
        if count_sum != 0:
            df_prob_table.loc[df_prob_table['0'] == str(prereq_grade), 'count'] *= 1/count_sum

    df_prob_table.rename(columns={'count': 'probability'}, inplace=True)

    return df_prob_table


# Creates a DataFrame similar to the final CPT structure with a column of averages of grades instead of probabilities
# This gives a slight preference towards higher grades due to rounding.
def find_averages(df_cpt_structure):
    df_averages = df_cpt_structure.astype(float)
    df_averages['Average'] = df_averages.mean(axis=1)
    df_averages['Average'] = df_averages['Average'].round()

    return df_averages.astype(str)
