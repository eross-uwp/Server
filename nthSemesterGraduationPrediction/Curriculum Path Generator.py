import pandas as pd
import numpy as np
import urllib.request

TRAIN_DATA_PATH = 'https://raw.githubusercontent.com/earos-uwp/Server/master/nthSemesterGraduationPrediction/' \
                  'data/Curriculum%20Structure.csv'
POSTREQ = 'postreq'
PREREQ = 'prereq'
RELATION = 'relationship'

endOfPath = 0

url = TRAIN_DATA_PATH
file = urllib.request.urlopen(url)
curriculumStructure = pd.read_csv(file)  # Getting curriculum Structure dataset from Github

postReqCourse = curriculumStructure[POSTREQ].values
prereq = curriculumStructure[PREREQ].values
relation = curriculumStructure[RELATION].values

allCourseWithPrereq = pd.DataFrame()


def findPrereqs(courseArray):
    courseIndexes = np.where(postReqCourse == courseArray[-1])
    for courseIndex in courseIndexes:
        if relation[courseIndex.any] == 'single':
            courseArray = courseArray.append(prereq[courseIndex])
            findPrereqs(courseArray)
        if relation[courseIndex.any()] == 'or':

        if relation[courseIndex.any()] == 'and':

     yield courseArray



if __name__ == "__main__":
    for course in np.unique(postReqCourse):
        allCourseWithPrereq = allCourseWithPrereq.insert(0, str(course), [], 1)
    tempArray = np.array()
    for course in np.unique(postReqCourse):
