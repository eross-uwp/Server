import seaborn; seaborn.set_style('whitegrid')
from pomegranate import *

from bayesian_network.Practice.OOPS2Practice.CSVReadWrite import *

# Returns a Discrete Distribution by formatting input course data
# Each element of CourseData is a grade probability
# These probabilities are overwritten with fitting data to the model
def getFormattedDiscDist(courseData):
    return DiscreteDistribution({'10': float(courseData[1]), '9': float(courseData[2]),
                                 '8': float(courseData[3]), '7': float(courseData[4]), '6': float(courseData[5]),
                                 '5': float(courseData[6]), '4': float(courseData[7]), '3': float(courseData[8]),
                                 '2': float(courseData[9]), '1': float(courseData[10]), '0': float(courseData[11])})

# Returns a touple of list of Discrete Distributions and
# a list of the States of the courses that are not dependent on other courses
# States (Nodes) are grade probabilities combined with a course name
def createDiscDistLists(dataFile):
    allData = getDiscDistData(dataFile)

    discDistList = []
    stateList = []

    for courseData in allData:
        discDist = getFormattedDiscDist(courseData)
        discDistList.append(discDist)
        state = State(discDist, courseData[0])
        stateList.append(state)

    return discDistList, stateList