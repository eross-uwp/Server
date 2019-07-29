from bayesian_network import BayesianNetwork
from knowledge_base import KnowledgeBase

if __name__ == "__main__":
    knowledge_base = KnowledgeBase('..\\..\\Data\\combined_course_structure.csv', '..\\ExcelFiles\\courses_and_grades.csv')
    bayes_net = BayesianNetwork(knowledge_base)

    bayes_net.update_cpt_tables()
    test = bayes_net.get_graph().get_node('Calculus and Analytic Geometry I').get_cp_table().get_table()

    print(test)
