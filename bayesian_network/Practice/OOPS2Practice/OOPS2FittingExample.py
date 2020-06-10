from pomegranate import *
import seaborn; seaborn.set_style('whitegrid')

from bayesian_network.Practice.OOPS2Practice.NodeCreator import *
from bayesian_network.Practice.CSVDataReader import *

# The goal of this file was to see how fitting (training the network) with real data works and
# to create a network from data alone, instead of with predetermined probabilities.
# However, as known on Jun 9 2020, there is no way in pomegranate to create a network and train the probabilities
# from the data while also specifying a prereq dependency structure. So, to get around this, the following
# code gives temporary probability distributions so a prereq structure can be formed to then have the probability
# data overwritten with trained real data.

# Real OOPS2 practice data located on google drive at \Summer_2020\BayesianNetwork\PracticeSampleData

PREREQ_FILE = 'PracticeSampleData - OOPS2 Prereqs.csv'
COURSE_FILE = 'PracticeSampleData - OOPS2 Event.csv'
FIT_FILE = 'PracticeSampleData - OOPS2 Real Data.csv'

# Create the Bayesian network object with a the previous data
model = BayesianNetwork("OOPS2 Network")
print("Bayesian Network created \n")

# Creates discrete distributions and the states of courses with no course dependencies using course name and
# grade probabilities from csv
# Basically, creates a list of unlabeled prereq data and a list of nodes with labeled prereq data
courseDiscDistList, courseDiscDistStateList = createDiscDistLists(PREREQ_FILE)
print("Prereq courses created \n")

# Creates a conditional probability table for the target course with events read in from a csv and
# the list of prereq course grade probabilities it is dependent on
courseConProb = ConditionalProbabilityTable(getCourseEvents(COURSE_FILE), courseDiscDistList)

# Creates the node for the target course with the conditional probability table and a course name
# Currently hard coded name for this file, but it should be read from csv
courseState = State(courseConProb, name="OOPS2")
print("Target course created \n")

# Adds all the course nodes to the model
for state in courseDiscDistStateList:
    model.add_state(state)

model.add_state(courseState)
print("Course nodes added to the model \n")

# Add edges which represent conditional dependencies, where the second node is conditionally dependent on the first node
# Basically, tells the model that the courses are prereqs to the target course
for state in courseDiscDistStateList:
    model.add_edge(state, courseState)

print("Course dependencies added to the model \n")

# Finalizes the Bayesian Network structure
model.bake()
print("Bayesian Network structure finalized \n")

# Fitting using real OOPS2 grades and prereq grades
model.fit(getOOPS2FittingData())
print("Data fit complete \n")

print(model.structure)

