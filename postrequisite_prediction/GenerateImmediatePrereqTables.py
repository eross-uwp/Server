"""
___authors___: Evan Majerus and Austin FitzGerald
"""
import pandas as pd

from TreeScripts.TreeMaker import TreeMaker

__COMBINED_COURSE_STRUCTURE_FILEPATH = '..\\Data\\combined_course_structure.csv'
__STUDENT_GRADE_LIST_WITH_TERMS_FILEPATH = 'data\\student_grade_list_with_terms.csv'
__OUTPUT_CSV_FILEPATH = 'data\\ImmediatePrereqTables\\'


class GenerateImmediatePrereqTables:
    __STUDENT_ID = 'student_id'
    __CUMULATIVE_GPA = 'cumulative_gpa'
    __PREV_TERM_GPA = 'prev_term_gpa'
    __STRUGGLE = 'struggle'
    __TERM_DIFFERENCE = 'term_difference'
    __CUMULATIVE_GPA_FILEPATH = 'data\\cumulative_gpa.csv'
    __TERM_GPA_FILEPATH = 'data\\term_gpa.csv'
    __STRUGGLING_PER_TERM_FILEPATH = 'data\\struggling_per_term.csv'

    def create_data_frame(self, tree, grades):
        postrequisite = tree.get_name()
        data_frame = pd.DataFrame(columns=[self.__STUDENT_ID, postrequisite])
        prerequisite = tree.get_immediate_prereqs()
        for j in prerequisite:
            data_frame[j.get_name()] = ''
        data_frame[self.__CUMULATIVE_GPA] = ''
        data_frame[self.__PREV_TERM_GPA] = ''
        data_frame[self.__STRUGGLE] = ''
        data_frame[self.__TERM_DIFFERENCE] = ''
        if self.__check_course(postrequisite, grades):
            data_frame = self.__get_student_info(data_frame, grades, postrequisite, prerequisite)
        return data_frame

    def __get_student_info(self, data_frame, grades, postrequisite, prerequisite):
        data_frame_row = 1
        for j, tier in grades.iterrows():
            if grades.at[j, postrequisite] != '' and self.__taken_prereq(j, grades, prerequisite):
                data_frame.at[data_frame_row, self.__STUDENT_ID] = grades.at[j, self.__STUDENT_ID]
                data_frame.at[data_frame_row, postrequisite] = grades.at[j, postrequisite].split(',')[1]
                postreq_term = int(grades.at[j, postrequisite].split(',')[0])
                earliest_term = 2000
                for k in prerequisite:
                    if self.__check_course(k.get_name(), grades):
                        if grades.at[j, k.get_name()] != '':
                            data_frame.at[data_frame_row, k.get_name()] = grades.at[j, k.get_name()].split(',')[1]
                            if int(grades.at[j, k.get_name()].split(',')[0]) < earliest_term:
                                earliest_term = int(grades.at[j, k.get_name()].split(',')[0])
                data_frame.at[data_frame_row, self.__TERM_DIFFERENCE] = postreq_term - earliest_term
                data_frame = self.__get_cumulative_gpa(data_frame, data_frame_row, j, earliest_term)
                data_frame = self.__get_prev_term_gpa(data_frame, data_frame_row, j, earliest_term)
                data_frame = self.__have_struggled(data_frame, data_frame_row, j, earliest_term)
                data_frame_row = data_frame_row + 1
        return data_frame

    def __taken_prereq(self, index, grades, prerequisite):
        for k in prerequisite:
            if self.__check_course(k.get_name(), grades):
                if grades.at[index, k.get_name()] != '':
                    return True
        return False

    def __get_cumulative_gpa(self, data_frame, data_frame_row, id, term):
        cumulative = pd.read_csv(self.__CUMULATIVE_GPA_FILEPATH).fillna('')
        columns = list(cumulative)

        index = columns.index(str(term))  # starting index TODO Add - 1
        gpa_found = 0
        while index != 0 and gpa_found != 1:
            if cumulative.at[id, columns[index]] != '':
                gpa = cumulative.at[id, columns[index]]
                gpa_found = 1
            else:
                index -= 1
        if gpa_found == 0:
            gpa = '$'

        data_frame.at[data_frame_row, self.__CUMULATIVE_GPA] = gpa
        return data_frame

    def __get_prev_term_gpa(self, data_frame, data_frame_row, id, term):
        prev_term_gpa = pd.read_csv(self.__TERM_GPA_FILEPATH).fillna('')
        columns = list(prev_term_gpa)

        index = columns.index(str(term))  # starting index TODO Add - 1
        gpa_found = 0
        while index != 0 and gpa_found != 1:
            if prev_term_gpa.at[id, columns[index]] != '':
                gpa = prev_term_gpa.at[id, columns[index]]
                gpa_found = 1
            else:
                index -= 1
        if gpa_found == 0:
            gpa = '$'

        data_frame.at[data_frame_row, self.__PREV_TERM_GPA] = gpa
        return data_frame

    def __have_struggled(self, data_frame, data_frame_row, id, term):
        struggle_per_term = pd.read_csv(self.__STRUGGLING_PER_TERM_FILEPATH).fillna('')
        columns = list(struggle_per_term)

        index = columns.index(str(term))  # starting index TODO Add - 1
        struggle_found = 0
        while index != 0 and struggle_found != 1:
            if struggle_per_term.at[id, columns[index]] != '':
                struggle = struggle_per_term.at[id, columns[index]]
                struggle_found = 1
            else:
                index -= 1
        if struggle_found == 0:
            struggle = '$'

        data_frame.at[data_frame_row, self.__STRUGGLE] = struggle
        return data_frame

    def __check_course (self, course, grades):
        courses = list(grades)
        for j in courses:
            if course == j:
                return True
        return False


if __name__ == "__main__":
    structure = pd.read_csv(__COMBINED_COURSE_STRUCTURE_FILEPATH).fillna('')
    grades = pd.read_csv(__STUDENT_GRADE_LIST_WITH_TERMS_FILEPATH).fillna('')
    prerequisite_tree_maker = TreeMaker(__COMBINED_COURSE_STRUCTURE_FILEPATH)
    postreqquisite_lrm = GenerateImmediatePrereqTables()
    count = 1
    for i, row in structure.iterrows():
        print(count)
        tree = prerequisite_tree_maker.process(row['postreq'])
        data_frame = postreqquisite_lrm.create_data_frame(tree, grades)
        data_frame.to_csv(__OUTPUT_CSV_FILEPATH
                          + "".join([c for c in tree.get_name() if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
                          + '.csv', index=False)
        count = count + 1
    print("Done!")
