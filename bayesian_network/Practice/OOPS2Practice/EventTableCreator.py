import csv
from itertools import product

# Takes in the number of dependent variables and the number of possible integer values
# Ex/ OOPS2 has 3 prereqs, so depNum = 3 and grades range from 0-10 using the full scale, so valNum = 11
# Creates a list of rows with every possible event and an equal probability for each
def createEventTable(depNum, valNum):
    probability = 1/valNum
    dataCol = depNum + 1
    valueList = []

    for i in range(0, valNum):
        valueList.append(i)

    valueList.reverse()

    eventTable = []

    for row in product(valueList, repeat = dataCol):
        eventTable.append(row)

    eventTable = [list(ele) for ele in eventTable]

    for row in eventTable:
        for element in row:
            element = str(element)
        row.append(probability)
        print(row)

    return eventTable

