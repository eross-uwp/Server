from bayesian_network import BayesianNetwork
from knowledge_base import KnowledgeBase
from graph_builder import GraphBuilder
from acyclic_graph import AcyclicGraph

if __name__ == "__main__":
    _data_file_path = '..\\ExcelFiles\\courses_and_grades.csv'
    _relations_file_path = '..\\..\\Data\\combined_course_structure.csv'

    knowledge_base = KnowledgeBase(_relations_file_path, _data_file_path)

    builder = GraphBuilder()
    builder = builder.build_nodes(list(knowledge_base.get_data().columns))
    builder = builder.add_parents(knowledge_base.get_relations())
    builder = builder.add_children()
    builder = builder.build_edges()

    graph = builder.build_graph()

    bayes_net = BayesianNetwork(knowledge_base, graph)

    print(bayes_net.get_graph().get_node('Calculus and Analytic Geometry I').get_parents())


