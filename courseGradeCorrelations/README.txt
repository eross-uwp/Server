This experiment is to determine the correlation between grades in all combinations of courses from our dataset.
DataSetGenerator contains a function that finds all combinations (not permutations) of courses and outputs the 2 column
list to a csv. CalculateCorrelations then finds all students who took both courses and stores their grade for each,
once all grades for a particular set of courses have been found, Spearman's rank-order correlation coefficient process
(through scipy stats) is run. This returns a p-value and rho for each set, which is stored in a csv.

The script in CalculateCorrelations was used in Azure Machine Learning Studio in 12 Python script modules alongside
the 12-split combinations dataset from DataSetGenerator. This allowed for a much quicker computational time. That Azure
experiment can be viewed here: https://gallery.cortanaintelligence.com/Experiment/Course-Grade-Correlations-12-Split