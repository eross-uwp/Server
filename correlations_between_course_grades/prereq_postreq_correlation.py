"""
___authors___: Austin FitzGerald
"""

import pandas as pd

from TreeScripts.TreeMaker import TreeMaker

__COURSE_CORRELATIONS_DF = pd.read_csv('..\\Data\\course_correlations_bonferroni.csv')
__COMBINED_STRUCTURE = '..\\Data\\combined_course_structure.csv'
__COMBINED_STRUCTURE_DF = pd.read_csv(__COMBINED_STRUCTURE)
__OUTPUT_FILEPATH = 'results\\postreq_prereq_correlations.csv'


if __name__ == "__main__":
    postreq_list = []
    prereq_list = []
    relationship_list = []
    pval_list = []
    rho_list = []
    n_list = []

    tree_maker = TreeMaker(__COMBINED_STRUCTURE)
    for i, postreq_row in __COMBINED_STRUCTURE_DF.iterrows():
        postreq_node = tree_maker.process(postreq_row['postreq'])
        prereqs = postreq_node.get_immediate_prereqs()
        for j, prereq_node in enumerate(prereqs):
            try_1 = __COURSE_CORRELATIONS_DF.loc[(__COURSE_CORRELATIONS_DF['class_1'] == postreq_node.get_name()) & (__COURSE_CORRELATIONS_DF['class_2'] == prereq_node.get_name())]
            try_2 = __COURSE_CORRELATIONS_DF.loc[(__COURSE_CORRELATIONS_DF['class_2'] == postreq_node.get_name()) & (__COURSE_CORRELATIONS_DF['class_1'] == prereq_node.get_name())]

            if try_1.empty and not try_2.empty:
                postreq_list.append(postreq_node.get_name())
                prereq_list.append(prereq_node.get_name())
                relationship_list.append(prereq_node.get_relationship())
                pval_list.append(try_2['pval'].values[0])
                rho_list.append(try_2['rho'].values[0])
                n_list.append(try_2['n'].values[0])
            elif try_2.empty and not try_1.empty:
                postreq_list.append(postreq_node.get_name())
                prereq_list.append(prereq_node.get_name())
                relationship_list.append(prereq_node.get_relationship())
                pval_list.append(try_1['pval'].values[0])
                rho_list.append(try_1['rho'].values[0])
                n_list.append(try_1['n'].values[0])

    final_df = pd.DataFrame({'postreq':postreq_list, 'prereq':prereq_list, 'relationship':relationship_list, 'pval':pval_list, 'rho':rho_list, 'n':n_list})
    final_df.to_csv(__OUTPUT_FILEPATH, index=False)