import os
import sys

import csv

# Only reads data from a fake grade probabilities csv for now
def getData(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        allData = []

        for row in csv_reader:
            rowData = []

            if line_count == 0:
                print(f'{", ".join(row)}')
                line_count += 1
            else:
                for i in range(0, 13):
                    rowData.append(row[i])

                allData.append(rowData)
                # print(f'{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]}, {row[7]}, {row[8]}, {row[9]}, {row[10]}, {row[11]}, {row[12]}')
                print(rowData)
                line_count += 1

        print(f'Processed {line_count - 1} course grade probabilities. \n')

    return allData

# Formatted for OOPS2 practice data, but can be extended to be more general
def getOOPS2FittingData():
    with open('PracticeSampleData - OOPS2 Real Data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        fittingData = []

        for row in csv_reader:

            if line_count == 0:
                print(f'{", ".join(row)}')
                line_count += 1
            else:
                rowData = [f"{row[0]}", f"{row[1]}", f"{row[2]}", f"{row[3]}"]

                fittingData.append(rowData)
                print(f'{row[0]}, {row[1]}, {row[2]}, {row[3]}')
                line_count += 1

        print(f'Processed {line_count - 1} real OOPS2 course fitting data.\n')

    return fittingData


# Hard coded to only work with specific course6 practice data - changed to fit shortened data
# This would likely need to completely change for real data
def getCourseEvent():
    with open('PracticeSampleData - Fake Course6 Event Shortened.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        couse6Data = []

        for row in csv_reader:

            if line_count == 0:
                print(f'{", ".join(row)}')
                line_count += 1
            else:
                rowData = [f"{row[0]}", f"{row[1]}", f"{row[2]}", float(row[3])]

                couse6Data.append(rowData)
                # print(f'{row[0]}, {row[1]}, {row[2]}, {row[3]}')
                line_count += 1

        print(f'Processed {line_count - 1} course grade probabilities.\n')

    return couse6Data

# Hard coded to only work with specific course6 practice data
# This would likely need to completely change for real data
def getFittingData():
    with open('PracticeSampleData - Fake Fitting Data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        fittingData = []

        for row in csv_reader:

            if line_count == 0:
                print(f'{", ".join(row)}')
                line_count += 1
            else:
                rowData = [f"{row[0]}", f"{row[1]}", f"{row[2]}"]

                fittingData.append(rowData)
                # print(f'{row[0]}, {row[1]}, {row[2]}')
                line_count += 1

        print(f'Processed {line_count - 1} course fitting data.\n')

    return fittingData



