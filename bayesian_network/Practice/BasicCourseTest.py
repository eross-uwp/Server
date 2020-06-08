import os
import sys

import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import numpy

from pomegranate import *

from bayesian_network.Practice.DiscreteDistributionCreator import createDiscDistList
from bayesian_network.Practice.CSVDataReader import getCourseEvent

numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

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

# Add edges which represent conditional dependencies, where the second node is
# conditionally dependent on the first node
# This example makes an edge from each of the 5 loaded in courses to course6 - shortened version has 2 courses
for state in courseDiscDistStateList:
    model.add_edge(state, s6)

# Makes the Bayesian network do its job
model.bake()

# Prints out the prediction and then the probabilities of the None values given as many other values as possible.
# The more values given, the more accurate the Bayesian network can predict the rest of the values.
# Input the grades of the courses as the following:
# Course1, Course2, Course3, Course4, Course5, Course6 - shortened version has 2 courses
# print(model.predict_proba([['A', 'A', 'A', 'A', 'A', None]]))

print(model.predict([['A', 'A', None]]))
print("")
print(model.predict_proba([['A', 'A', None]]))
print("\n\n")

print(model.predict([['D+', None, 'D']]))
print("")
print(model.predict_proba([['D+', None, 'D']]))
print("\n\n")

print(model.predict([[None, None, 'A-']]))
print("")
print(model.predict_proba([[None, None, 'A-']]))
print("\n\n")

print(model.predict([[None, None, None]]))
print("")
print(model.predict_proba([[None, None, None]]))
print("\n\n")