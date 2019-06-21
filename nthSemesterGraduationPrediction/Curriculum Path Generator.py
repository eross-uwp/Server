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


if __name__ == "__main__":
    unique, counts = np.unique(relation, return_counts=True)
    total = dict(zip(unique, counts))
    print(total['or'])
    allCourseWithPrereq = pd.DataFrame(columns=range(np.unique(postReqCourse).size))#np.unique(postReqCourse), c

    print('dsf')
    print('dsf')
    while True:
        for eachPostreq in allCourseWithPrereq:
            courseIndexes = np.where(postReqCourse == eachPostreq.values[-1])
            for courseIndex in courseIndexes:
                if relation[courseIndex].any == 'single':
                    eachPostreq.values = eachPostreq.values.append(prereq[courseIndex])
                #if relation[courseIndex.any()] == 'or':
                #if relation[courseIndex.any()] == 'and':
        break
