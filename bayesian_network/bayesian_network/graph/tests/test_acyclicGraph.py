from unittest import TestCase
from acyclic_graph import AcyclicGraph
from node import Node
from edge import Edge


class TestAcyclicGraph(TestCase):
    def test_get_node(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        a.add_child(b)
        a.add_child(c)

        e1 = Edge(a, b)
        e2 = Edge(a, c)

        g = AcyclicGraph([a, b, c], [e1, e2])

        self.assertTrue(a == g.get_node('A'))
        self.assertTrue(g.get_node('Z') is None)

    def test_get_edge(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        a.add_child(b)
        a.add_child(c)

        e1 = Edge(a, b)
        e2 = Edge(a, c)

        g = AcyclicGraph([a, b, c], [e1, e2])

        self.assertTrue([e1, e2] == g.get_edge('A'))
        self.assertTrue([e2] == g.get_edge('C'))
        self.assertTrue(g.get_edge('Z') is None)

    def test_add_node(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')
        d = Node('D')

        a.add_child(b)
        a.add_child(c)

        e1 = Edge(a, b)
        e2 = Edge(a, c)

        nodes = [a, b, c]
        edges = [e1, e2]

        g = AcyclicGraph(nodes, edges)
        """
        A
        |-B
        ||-D
        |-C  
        """
        g.add_node([b], d)

        self.assertTrue(d == g.get_node('D'))
        e = g.get_edge('D')
        self.assertTrue(b == e[0].get_parent() and d == e[0].get_child())

        g1 = AcyclicGraph([a, c], [e2])
        b.add_child(d)
        g1.add_node([a], b)

        self.assertTrue(b == g1.get_node('B'))
        self.assertTrue(d == g1.get_node('D'))

        e = g1.get_edge('B')
        self.assertTrue(a == e[0].get_parent() and b == e[0].get_child())
        self.assertTrue(b == e[1].get_parent() and d == e[1].get_child())

    def test_get_parents(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')
        d = Node('D')

        a.add_child(b)
        a.add_child(c)
        d.add_child(b)
        e1 = Edge(a, b)
        e2 = Edge(a, c)
        e3 = Edge(d, b)

        g = AcyclicGraph([a, b, c], [e1, e2, e3])

        p = g.get_parents('B')

        self.assertTrue(2 == len(p))
        self.assertTrue(a == p[0])
        self.assertTrue(d == p[1])
        self.assertTrue(g.get_parents('A') is None)

    def test_add_edge(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        e1 = Edge(a, b)
        e2 = Edge(a, c)

        g = AcyclicGraph([a, b, c], [e1])

        g.add_edge(e2)

        self.assertTrue(e2 == g.get_edges()[1])

    def test_add_edges(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        g = AcyclicGraph([a, b, c])

        e1 = Edge(a, b)
        e2 = Edge(a, c)

        g.add_edges([e1, e2])
        e = g.get_edges()
        self.assertTrue(e1 == e[0] and e2 == e[1])
        self.assertTrue(2 == len(g.get_edges()))

    def test_remove_edge(self):

        a = Node('A')
        b = Node('B')
        c = Node('C')

        a.add_child(b)
        a.add_child(c)

        e1 = Edge(a, b)
        e2 = Edge(a, c)

        g = AcyclicGraph([a, b, c], [e1, e2])
        g.remove_edge('A', 'B')
        e = g.get_edges()
        self.assertTrue(1 == len(e))
        self.assertTrue(e2 == e[0])

        n = g.get_node('A')

        self.assertTrue(n.get_child('B') is None)

    def test_remove_node(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        a.add_child(b)
        a.add_child(c)

        e1 = Edge(a, b)
        e2 = Edge(a, c)

        g = AcyclicGraph([a, b, c], [e1, e2])
        g.remove_node('B')

        self.assertTrue(1 == len(g.get_edges()))
        self.assertTrue(g.get_node('B') is None)

        n = g.get_node('A')

        self.assertTrue(n.get_child('B') is None)


