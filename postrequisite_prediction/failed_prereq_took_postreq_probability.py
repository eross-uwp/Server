"""
___authors___: Austin FitzGerald
"""

from numpy.core.defchararray import upper

from TreeScripts.TreeMaker import TreeMaker
import pandas as pd

__COMBINED_STRUCTURE = 'data\\combined_structure.csv'
__COMBINED_STRUCTURE_DATAFRAME = pd.read_csv(__COMBINED_STRUCTURE)
__GRADES_WITH_TERM_DATAFRAME_ = pd.read_csv('data\\student_grade_list_with_terms.csv').fillna('')
__OUTPUT_FILEPATH = 'results\\failed_prereq_took_postreq_probability.csv'


def convert_grade(string_grade):
    string_grade = upper(string_grade)
    if string_grade == 'A':
        return 10
    elif string_grade == 'A-':
        return 9
    elif string_grade == 'B+':
        return 8
    elif string_grade == 'B':
        return 7
    elif string_grade == 'B-':
        return 6
    elif string_grade == 'C+':
        return 5
    elif string_grade == 'C':
        return 4
    elif string_grade == 'C-':
        return 3
    elif string_grade == 'D+':
        return 2
    elif string_grade == 'D':
        return 1
    elif string_grade == 'F':
        return 0
    else:
        return 0


def append_if_division_allowed(append_to, divide_this, by_this):
    if by_this != 0:
        append_to.append(divide_this / by_this)
    else:
        append_to.append(0)


if __name__ == "__main__":
    postreq_list = []
    tree_maker = TreeMaker(__COMBINED_STRUCTURE)

    prereq_failed_postreq_taken_list = []
    prereq_failed_postreq_not_taken_list = []
    prereq_failed_postreq_passed_list = []
    prereq_failed_postreq_passed_probability = []
    prereq_failed_postreq_failed_list = []
    prereq_failed_postreq_taken_probability = []

    prereq_passed_postreq_taken_list = []
    prereq_passed_postreq_not_taken_list = []
    prereq_passed_postreq_passed_list = []
    prereq_passed_postreq_passed_probability = []
    prereq_passed_postreq_failed_list = []
    prereq_passed_postreq_taken_probability = []

    for i, postreq_row in __COMBINED_STRUCTURE_DATAFRAME.iterrows():
        prereq_failed_postreq_taken_list.append(0)
        prereq_failed_postreq_not_taken_list.append(0)
        prereq_failed_postreq_passed_list.append(0)
        prereq_failed_postreq_failed_list.append(0)

        prereq_passed_postreq_taken_list.append(0)
        prereq_passed_postreq_not_taken_list.append(0)
        prereq_passed_postreq_passed_list.append(0)
        prereq_passed_postreq_failed_list.append(0)

        postreq_with_children = tree_maker.process(postreq_row['postreq'])
        postreq_list.append(postreq_with_children.get_name())
        prereqs = postreq_with_children.get_immediate_prereqs()
        for idx, student_row_series in __GRADES_WITH_TERM_DATAFRAME_.iterrows():
            prereq_failed_postreq_taken = 0
            prereq_failed_postreq_not_taken = 0
            prereq_failed_postreq_passed = 0
            prereq_failed_postreq_failed = 0

            prereq_passed_postreq_taken = 0
            prereq_passed_postreq_not_taken = 0
            prereq_passed_postreq_passed = 0
            prereq_passed_postreq_failed = 0

            prereq_failed_took_postreq_prob = 0
            if postreq_with_children.get_name() in __GRADES_WITH_TERM_DATAFRAME_:
                postreq_term_and_grade = __GRADES_WITH_TERM_DATAFRAME_[postreq_with_children.get_name()].values[idx]
                for j, prereq in enumerate(prereqs):
                    prereq_grade_req = convert_grade(prereq.get_grade())
                    if prereq.get_name() in __GRADES_WITH_TERM_DATAFRAME_:
                        prereq_grade = __GRADES_WITH_TERM_DATAFRAME_[prereq.get_name()].values[idx]
                        if prereq_grade != '':  # took prereq
                            prereq_grade = convert_grade(prereq_grade.split(',')[1])
                            if prereq_grade <= prereq_grade_req:  # failed prereq
                                if postreq_term_and_grade != '':  # took postreq
                                    prereq_failed_postreq_taken += 1
                                    if convert_grade(postreq_term_and_grade.split(',')[1]) >= convert_grade(
                                            postreq_with_children.get_grade()):  # postreq passed
                                        prereq_failed_postreq_passed += 1
                                    else:
                                        prereq_failed_postreq_failed += 1
                                elif postreq_term_and_grade == '':  # did not take postreq
                                    prereq_failed_postreq_not_taken += 1
                            else:  # passed prereq
                                if postreq_term_and_grade != '':  # took postreq
                                    prereq_passed_postreq_taken += 1
                                    if convert_grade(postreq_term_and_grade.split(',')[1]) >= convert_grade(
                                            postreq_with_children.get_grade()):  # postreq passed
                                        prereq_passed_postreq_passed += 1
                                    else:
                                        prereq_passed_postreq_failed += 1
                                elif postreq_term_and_grade == '':  # did not take postreq
                                    prereq_passed_postreq_not_taken += 1
            if prereq_failed_postreq_taken > 0:
                prereq_failed_postreq_taken_list[i] += 1
            if prereq_failed_postreq_not_taken > 0:
                prereq_failed_postreq_not_taken_list[i] += 1
            if prereq_failed_postreq_passed > 0:
                prereq_failed_postreq_passed_list[i] += 1
            if prereq_failed_postreq_failed > 0:
                prereq_failed_postreq_failed_list[i] += 1
            if prereq_passed_postreq_taken > 0:
                prereq_passed_postreq_taken_list[i] += 1
            if prereq_passed_postreq_not_taken > 0:
                prereq_passed_postreq_not_taken_list[i] += 1
            if prereq_passed_postreq_passed > 0:
                prereq_passed_postreq_passed_list[i] += 1
            if prereq_passed_postreq_failed > 0:
                prereq_passed_postreq_failed_list[i] += 1
        append_if_division_allowed(prereq_failed_postreq_passed_probability, prereq_failed_postreq_passed_list[i],
                               prereq_failed_postreq_taken_list[i] + prereq_failed_postreq_not_taken_list[i])
        append_if_division_allowed(prereq_failed_postreq_taken_probability, prereq_failed_postreq_taken_list[i],
                                   prereq_failed_postreq_taken_list[i] + prereq_failed_postreq_not_taken_list[i])
        append_if_division_allowed(prereq_passed_postreq_passed_probability, prereq_passed_postreq_passed_list[i],
                               prereq_passed_postreq_taken_list[i] + prereq_passed_postreq_not_taken_list[i])
        append_if_division_allowed(prereq_passed_postreq_taken_probability, prereq_passed_postreq_taken_list[i],
                                   prereq_passed_postreq_taken_list[i] + prereq_passed_postreq_not_taken_list[i])

    final_dataframe = pd.DataFrame(
        {'Postreq': postreq_list, 'prereq_failed_postreq_taken': prereq_failed_postreq_taken_list,
         'prereq_failed_postreq_not_taken': prereq_failed_postreq_not_taken_list,
         'prereq_failed_postreq_passed': prereq_failed_postreq_passed_list,
         'prereq_failed_postreq_passed_probability': prereq_failed_postreq_passed_probability,
         'prereq_failed_postreq_failed': prereq_failed_postreq_failed_list,
         'prereq_failed_postreq_taken_probability': prereq_failed_postreq_taken_probability,
         'prereq_passed_postreq_taken': prereq_passed_postreq_taken_list,
         'prereq_passed_postreq_not_taken': prereq_passed_postreq_not_taken_list,
         'prereq_passed_postreq_passed': prereq_passed_postreq_passed_list,
         'prereq_passed_postreq_passed_probability': prereq_passed_postreq_passed_probability,
         'prereq_passed_postreq_failed': prereq_passed_postreq_failed_list,
         'prereq_passed_postreq_taken_probability': prereq_passed_postreq_taken_probability})
    final_dataframe.to_csv(__OUTPUT_FILEPATH, index=False)
