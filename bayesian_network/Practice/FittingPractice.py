import os
import sys

import matplotlib.pyplot as plt
import seaborn;

seaborn.set_style('whitegrid')
import numpy

from pomegranate import *

from bayesian_network.Practice.DiscreteDistributionCreator import createDiscDistList
from bayesian_network.Practice.CSVDataReader import *

numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

# The goal of this file was to see how fitting (training the network) works and what the results look like
# It is almost identical to PredictionPractice, which gives more information

# Editable fake data located on google drive at \Summer_2020\BayesianNetwork\PracticeSampleData


# Creates discrete distribution and the states of courses with no course dependencies using course name and
# grade probabilities from csv
# Currently, the test csv has 5 of these courses - shortened version has 2 courses
courseDiscDistList, courseDiscDistStateList = createDiscDistList()

# List of probability events loaded from csv for course6 using fake event probabilities
course6_events = getCourseEvent()

# Course6 is dependent on course the 5 loaded in courses in courseDiscDistList - shortened version has 2 courses
course6 = ConditionalProbabilityTable(course6_events, courseDiscDistList)

# State objects hold both the distribution, and a high level name.
# In this case, course6 is dependent on other courses, so it uses a conditional probability
# instead of a discrete distribution of probabilities
s6 = State(course6, name="Course6")

# Create the Bayesian network object with a useful name
model = BayesianNetwork("ExampleCourses")

# Add the course states to the Bayesian network
for state in courseDiscDistStateList:
    model.add_state(state)

model.add_state(s6)

# Add edges which represent conditional dependencies, where the second node is conditionally
# dependent on the first node
# This example makes an edge from each of the 5 loaded in courses to course6 - shortened version has 2 courses
for state in courseDiscDistStateList:
    model.add_edge(state, s6)

# Finalizes the Bayesian Network structure
model.bake()

# Prints out predictions before training, these are based on random probabilities in course6 events csv
print("Before training, using random values")

print(model.predict([['B-', 'C+', None]]))
print("")
print(model.predict_proba([['B-', 'C+', None]]))
print("\n\n")

print(model.predict([[None, None, None]]))
print("")
print(model.predict_proba([[None, None, None]]))
print("\n\n")

# Trains the network with csv data
model.fit(getFittingData())

print("After training, using semi realistic values")

# Prints predictions after training, to compare to previous predictions
print(model.predict([['B-', 'C+', None]]))
print("")
print(model.predict_proba([['B-', 'C+', None]]))
print("\n\n")

print(model.predict([[None, None, None]]))
print("")
print(model.predict_proba([[None, None, None]]))
print("\n\n")

# Prints the probabilities for each grade in course1 and course2, then the probabilities for each event in course6
print(courseDiscDistList[0])
print("\n\n")

print(courseDiscDistList[1])
print("\n\n")

print(course6)
print("\n\n")