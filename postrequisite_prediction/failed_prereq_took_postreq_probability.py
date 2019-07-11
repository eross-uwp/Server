import copy

from numpy.core.defchararray import upper

from TreeScripts.TreeMaker import TreeMaker
import pandas as pd

__COMBINED_STRUCTURE = 'data\\combined_structure.csv'
__COMBINED_STRUCTURE_DATAFRAME = pd.read_csv(__COMBINED_STRUCTURE)
__GRADES_WITH_TERM_DATAFRAME_ = pd.read_csv('data\\student_grade_list_with_terms.csv').fillna('')


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


if __name__ == "__main__":
    tree_maker = TreeMaker(__COMBINED_STRUCTURE)
    for i, postreq_row in __COMBINED_STRUCTURE_DATAFRAME.iterrows():
        prereq_failed_but_took_postreq = 0
        prereq_failed_did_not_postreq = 0
        prereq_failed_postreq_passed = 0
        prereq_passed = 0

        postreq_with_children = tree_maker.process(postreq_row['postreq'])
        prereqs = postreq_with_children.get_immediate_prereqs()
        for idx, student_row_series in __GRADES_WITH_TERM_DATAFRAME_.iterrows():
            if postreq_with_children.get_name() in __GRADES_WITH_TERM_DATAFRAME_:
                postreq_term_and_grade = __GRADES_WITH_TERM_DATAFRAME_[postreq_with_children.get_name()].values[idx]
                for j, prereq in enumerate(prereqs):
                    prereq_grade_req = convert_grade(prereq.get_grade())
                    if prereq.get_name() in __GRADES_WITH_TERM_DATAFRAME_:
                        prereq_grade = __GRADES_WITH_TERM_DATAFRAME_[prereq.get_name()].values[idx]
                        if prereq_grade != '':  # took prereq
                            prereq_grade = convert_grade(prereq_grade.split(',')[1])
                            if prereq_grade <= prereq_grade_req:
                                if postreq_term_and_grade != '':
                                    prereq_failed_but_took_postreq += 1
                                    if convert_grade(postreq_term_and_grade.split(',')[1]) >= convert_grade(postreq_with_children.get_grade()):
                                        prereq_failed_postreq_passed += 1
                                elif postreq_term_and_grade == '':
                                    prereq_failed_did_not_postreq += 1
                            else:
                                prereq_passed += 1
        print('Postreq: ' + postreq_with_children.get_name() + '\n' +
              'Prereq failed first attempt but took postreq : ' + str(prereq_failed_but_took_postreq) + '\n' +
              '     - passed postreq: ' + str(prereq_failed_postreq_passed) + '\n' +
              'Prereq failed first attempt but did not take postreq: ' + str(prereq_failed_did_not_postreq) + '\n')

