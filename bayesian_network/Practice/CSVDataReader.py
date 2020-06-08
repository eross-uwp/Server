import os
import sys

import csv

# Only reads data from a fake grade probabilities csv for now
def getData():
    with open('PracticeSampleData - Fake Grade Probabilities Shortened.csv') as csv_file:
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
                print(
                    f'{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]}, {row[7]}, {row[8]}, {row[9]}, {row[10]}, {row[11]}, {row[12]}')
                line_count += 1

        print(f'Processed {line_count - 1} course grade probabilities. \n')

    return allData


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
                print(
                    f'{row[0]}, {row[1]}, {row[2]}, {row[3]}')
                line_count += 1

        print(f'Processed {line_count - 1} course grade probabilities.\n')

    return couse6Data



