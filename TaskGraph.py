import random
from itertools import repeat

class TaskGraph:

    def __init__(self, nodes : list[str], edges : tuple, bias : str):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.node_to_index = {}
        for i in range(len(self.nodes)):
            self.node_to_index[self.nodes[i]] = i
        self.bias = bias
        
        self.adj_matrix = [[] for i in repeat(10000, self.num_nodes)]
        for start_node, end_node, edge_weight in edges:
            self.adj_matrix[self._node_to_index(start_node)][self._node_to_index(end_node)] = edge_weight

    def _node_to_index(self, node):
        return self.node_to_index[node]

    #present, variable bias
    def _calc_variable_bias(self):
        return random.uniform(0, 1)

    def delete_edges(self):
        pass

    def traverse_optimal(self, start : str, goal : str):
        pass

    def traverse_procrastination(self, start : str, goal : str):
        pass


if __name__ == "__main__":
    pass