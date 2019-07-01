This experiment is to determine the correlation between grades in all combinations of courses from our dataset.
DataSetGenerator contains a function that finds all combinations (not permutations) of courses and outputs the 2 column
list to a csv. CalculateCorrelations then finds all students who took both courses and stores their grade for each,
once all grades for a particular set of courses have been found, Spearman's rank-order correlation coefficient process
(through scipy stats) is run. This returns a p-value and rho for each set, which is stored in a csv.

The fill script in CalculateCorrelations was used in Azure Machine Learning Studio in 12 Python script modules alongside
the 12-split combinations dataset from DataSetGenerator. This allowed for a much quicker computational time. That Azure
experiment can be viewed here: https://gallery.cortanaintelligence.com/Experiment/Course-Grade-Correlations-12-Split

CalculateCorrelations also contains a function to generate bubble scatter plots for all course combinations with a
n value (number of students who took both courses) that is greater or equal than a given parameter. The axis are the
grades for course one (x) and course two (y), the bubble size is determined by the frequency of the grade combination.