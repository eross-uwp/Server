from conditional_probability_table_builder import CPTBuilder
from node import Node
import itertools
import pandas as pd

# Todo: Need to implement how to get the probability when their is not enough data.
"""
When there isn't enough data to make a accurate prediction, we will weigh each grades then normalized the probability.
See Adjusted_CPT_Tables.xlsx for more details. In the method filter_data there is a condition that needs to be changes. 
This right condition is next to the if statement and is commented out. 
"""


class ConditionalProbabilityTable:
    def __init__(self, table):
        """
        Constructs a Conditional Probability Table
        :param node: Node
        """
        self._cpt = table

    def get_table(self):
        return self._cpt

    def update_table(self, node, kb):
        builder = CPTBuilder(node.get_name(), node.get_parent_names())
        self._cpt = builder.build(kb.get_data(), kb.get_scale())


