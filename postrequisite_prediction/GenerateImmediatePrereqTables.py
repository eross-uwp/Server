"""
___authors___: Evan Majerus and Austin FitzGerald
Script that retrieves a student's grade for each post requisite course and each immediate prerequisite course
they have taken and creates a csv with that inside.  It also gathers the term before the earliest prerequisite course
to get the cumulative GPA in that term and the term GPA.  It also saves the difference between the term of the
postrequisite and the prerequisite.  If a student did not take both the postrequisite and at least one of the
prerequisites they are not added to the csv.
"""
import pandas as pd

from postrequisite_prediction.TreeScripts.TreeMaker import TreeMaker

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

    """
    Creates a dataframe titled by the postrequisite course and with column headers of student id, postrequisite course
    name, all the prerequisite course names, cumulative gpa, previous term gpa, struggle, and term difference.
    Parameters: tree, grades
    Returns: data_frame
    """
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

    """
    Method that retrieves all the information for each student that has taken the postrequisite and at least one
    prerequisite for that postrequisite.  It then files the student information under the correct column headings in
    the data frame.
    Parameters: data_frame, grades, postrequisite, prerequisite
    Returns: data_frame
    """
    def __get_student_info(self, data_frame, grades, postrequisite, prerequisite):
        data_frame_row = 1
        for j, tier in grades.iterrows():
            if grades.at[j, postrequisite] != '' and self.__taken_prereq(j, grades, prerequisite):
                postreq_term = int(grades.at[j, postrequisite].split(',')[0])
                taken_prerequiste = self.__get_taken_prereq(j, grades, postreq_term, prerequisite)
                if len(taken_prerequiste)!= 0:
                    data_frame.at[data_frame_row, self.__STUDENT_ID] = grades.at[j, self.__STUDENT_ID]
                    data_frame.at[data_frame_row, postrequisite] = convert_grade(grades.at[j, postrequisite].split(',')[1])
                    earliest_term = postreq_term + 1
                    for k in taken_prerequiste:
                        if self.__check_course(k.get_name(), grades):
                            if grades.at[j, k.get_name()] != '':
                                data_frame.at[data_frame_row, k.get_name()] = convert_grade(grades.at[j, k.get_name()].split(',')[1])
                                if int(grades.at[j, k.get_name()].split(',')[0]) < earliest_term:
                                    earliest_term = int(grades.at[j, k.get_name()].split(',')[0])
                    data_frame.at[data_frame_row, self.__TERM_DIFFERENCE] = postreq_term - earliest_term
                    data_frame = self.__get_cumulative_gpa(data_frame, data_frame_row, j, earliest_term)
                    data_frame = self.__get_prev_term_gpa(data_frame, data_frame_row, j, earliest_term)
                    data_frame = self.__have_struggled(data_frame, data_frame_row, j, earliest_term)
                    data_frame_row = data_frame_row + 1
        return data_frame

    """
    Method that checks if a student has taken at least one prerequisite course for the postrequisite.
    Parameters: index, grades, prerequisite
    Returns: True, False
    """
    def __taken_prereq(self, index, grades, prerequisite):
        for k in prerequisite:
            if self.__check_course(k.get_name(), grades):
                if grades.at[index, k.get_name()] != '':
                    return True
        return False

    """
    Method that retrieves the prerequisite courses that the student has taken.
    Parameters: index, grades, postreq_term, prerequisite
    Returns: taken_prerequisite
    """
    def __get_taken_prereq(self, index, grades, postreq_term, prerequisite):
        taken_prerequisite = []
        for k in prerequisite:
            if self.__check_course(k.get_name(), grades):
                if grades.at[index, k.get_name()] != '':
                    if int(grades.at[index, k.get_name()].split(',')[0]) <= postreq_term:
                        taken_prerequisite.append(k)
        return taken_prerequisite


    """
    Method that gets teh cumulative gpa of the student the semester before they took the earliest prerequisite course.
    Parameters: data_frame, data_frame_row, id, term
    Returns: data_frame
    """
    def __get_cumulative_gpa(self, data_frame, data_frame_row, id, term):
        cumulative = pd.read_csv(self.__CUMULATIVE_GPA_FILEPATH).fillna('')
        columns = list(cumulative)

        index = columns.index(str(term))  # starting index
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

    """
    Method that gets the term gpa of the term right before the student took the earliest prerequisite course.
    Parameters: data_frame, data_frame_row, id, term
    Returns: data_frame
    """
    def __get_prev_term_gpa(self, data_frame, data_frame_row, id, term):
        prev_term_gpa = pd.read_csv(self.__TERM_GPA_FILEPATH).fillna('')
        columns = list(prev_term_gpa)

        index = columns.index(str(term))  # starting index
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

    """
    Method that checks a student's struggling status up to the earliest prerequisite
    Parameters: data_frame, data_frame_row, id, term
    Returns: data_frame
    """
    def __have_struggled(self, data_frame, data_frame_row, id, term):
        struggle_per_term = pd.read_csv(self.__STRUGGLING_PER_TERM_FILEPATH).fillna('')
        columns = list(struggle_per_term)

        index = columns.index(str(term))  # starting index
        struggle_found = 0
        while index != 0 and struggle_found != 1:
            if struggle_per_term.at[id, columns[index]] != '':
                struggle = convert_struggle(struggle_per_term.at[id, columns[index]])
                struggle_found = 1
            else:
                index -= 1
        if struggle_found == 0:
            struggle = '$'

        data_frame.at[data_frame_row, self.__STRUGGLE] = struggle
        return data_frame

    """
    Method that checks if a course is in the list of taken and existing courses.
    Parameters: course, grades
    Returns: True, False
    """
    def __check_course (self, course, grades):
        courses = list(grades)
        for j in courses:
            if course == j:
                return True
        return False

"""
Method that converts the stuggling strings into their associated numeric values
Parameters: string_struggle
Returns: 1, 2, 3
"""
def convert_struggle(string_struggle):
    if string_struggle == 'G':
        return 3
    elif string_struggle == 'S':
        return 2
    elif string_struggle == 'E':
        return 1

"""
Method that converts a string grade into it's associated numeric value.
Parameters: string_grade
Returns: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
"""
def convert_grade(string_grade):
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
                          + "".join([c for c in tree.get_name() if c.isalpha() or c.isdigit() or c == ' ' or c == '-'])
                          .rstrip()
                          + '.csv', index=False)
        count = count + 1
    print("Done!")
