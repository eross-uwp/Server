import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import numpy
numpy.random.seed(0)
numpy.set_printoptions(suppress=True)
import pandas as pd

from pomegranate import *
if __name__ == "__main__":
 rawdata = pd.read_csv('BNtestdataset.csv')

 course1dict = {}
 course2dict = {}
 course3dict = {}

 course1_grade_probs = rawdata.groupby('course1').size().div(len(rawdata))
 for x in range(len(course1_grade_probs)):
  course1dict[course1_grade_probs.index[x]] = course1_grade_probs.values[x]
 course1_distribution = DiscreteDistribution(course1dict)

 course2_grade_probs = rawdata.groupby('course2').size().div(len(rawdata))
 for x in range(len(course2_grade_probs)):
  course2dict[course2_grade_probs.index[x]] = course2_grade_probs.values[x]

 course2_distribution = DiscreteDistribution(course2dict)


 course3_grade_probs = rawdata.groupby('course3').size().div(len(rawdata))
 overall_grade_probs = rawdata.groupby(['course3', 'course2', 'course1']).size().div(rawdata.groupby(['course2', 'course1']).size())
 overall_grade_probs.index
 #overall_grade_probs.index = str(overall_grade_probs.index)
 print(overall_grade_probs)
 course3_cpt = []
 for each_3_grade in course3_grade_probs.index:
  for each_2_grade in course2_grade_probs.index:
   for each_1_grade in course1_grade_probs.index:
    print('a')
    try:
     temp = [each_2_grade, each_1_grade, each_3_grade, overall_grade_probs[each_2_grade, each_1_grade, each_3_grade]]
     course3_cpt.append(temp)
    except:
     temp = [each_2_grade, each_1_grade, each_3_grade, 0.0]
     course3_cpt.append(temp)


 course3_distribution = ConditionalProbabilityTable(course3_cpt, [course1_distribution, course2_distribution])

 s1 = Node(course1_distribution, name="course1")
 s2 = Node(course2_distribution, name="course2")
 s3 = Node(course3_distribution, name="course3")

 model = BayesianNetwork("test")
 model.add_states(s3, s2, s1)
 model.add_edge(s1, s3)
 model.add_edge(s2, s3)
 model.bake()

 print('\n')
 print(model.probability(['A', 'A', 'A']))
 print(model.probability(['A', 'B', 'B']))

