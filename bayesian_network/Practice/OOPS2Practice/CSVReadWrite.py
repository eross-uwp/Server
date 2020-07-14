import csv

# Returns formatted prereq probability data for use in discrete distribution and node creation
# The probabilities will be overwritten when the data is fit to the model
def getDiscDistData(dataFile):
    with open(dataFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        allData = []

        for row in csv_reader:
            rowData = []

            if line_count == 0:
                print(f'{", ".join(row)}')
                line_count += 1
            else:
                for i in range(0, 12):
                    rowData.append(row[i])

                allData.append(rowData)
                print(rowData)
                line_count += 1

        print(f'Processed {line_count - 1} prereq course grade probabilities. \n')

    return allData


# Returns a list of event data for the conditional probability table of the target predicted course
# Reads into the format [prereq1 grade, prereq2 grade, ... prereqN grade, target grade, probability]
# The probabilities will be overwritten when the data is fit to the model
def getCourseEvents(dataFile):
    with open(dataFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        eventData = []

        for row in csv_reader:

            if line_count == 0:
                print(f'{", ".join(row)}')
                line_count += 1
            else:
                rowData = []
                i = 0
                while not isCSVProbability(row,i):
                    rowData.append(f"{row[i]}")
                    i += 1

                rowData.append(float(row[i]))

                eventData.append(rowData)
                print(rowData)
                line_count += 1

        print(f'Processed {line_count - 1} target course events.\n')

    return eventData

# Returns true if read column value is the last item in the row for course events, this would be the probability
# Used for getCourseEvents()
def isCSVProbability(row, i):
    try:
        temp = row[i+1]
    except:
        return True

    return False
