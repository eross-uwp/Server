"""
__Author__: Nick Tiede

__Purpose__: To create a pomegranate state (node) list, whose probabilities will be overwritten from training,
             for each of the prereqs
"""
import pandas as pd
from pomegranate.base import State
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution


# Returns a touple of lists of generated discrete distributions and states (nodes) for each prereq
def create_disc_dist_state_list(course_names, num_prereqs, num_grades):
    disc_dist_state_list = []

    for i in range(0, num_prereqs):
        disc_dist = create_disc_dist(num_grades)
        disc_dist_state_list.append(State(disc_dist, course_names[i]))

    return disc_dist_state_list


# Returns a pomegranate discrete distribution with a generated dictionary based on number of grades
def create_disc_dist(num_grades):
    keys = list(map(str, range(0, num_grades)))
    values = [1/num_grades] * num_grades
    disc_dist_dict = dict(zip(keys, values))

    return DiscreteDistribution(disc_dist_dict)
