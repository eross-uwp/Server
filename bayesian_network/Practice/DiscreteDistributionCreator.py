import os
import sys

import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import numpy

from pomegranate import *

from Practice.CSVDataReader import getData

# Returns a Discrete Distribution by formatting input course data
def getFormattedDiscDist(courseData):
    return DiscreteDistribution({'A': float(courseData[1]), 'A-': float(courseData[2]),
                                 'B+': float(courseData[3]), 'B': float(courseData[4]), 'B-': float(courseData[5]),
                                 'C+': float(courseData[6]), 'C': float(courseData[7]), 'C-': float(courseData[8]),
                                 'D+': float(courseData[9]), 'D': float(courseData[10]), 'D-': float(courseData[11]),
                                 'F': float(courseData[12])})

# Returns a touple of list of Discrete Distributions and
# a list of the States of the courses that are not dependent on other courses
def createDiscDistList():
    allData = getData('PracticeSampleData - Fake Grade Probabilities Shortened.csv')

    discDistList = []
    stateList = []

    for courseData in allData:
        discDist = getFormattedDiscDist(courseData)
        discDistList.append(discDist)
        state = State(discDist, courseData[0])
        stateList.append(state)

    return discDistList, stateList
