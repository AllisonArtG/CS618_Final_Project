import random
import math
from copy import deepcopy

class TaskGraph:

    def __init__(self, nodes : list[str], edges : list[tuple], bias : str):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.node_to_index = {}
        for i in range(len(self.nodes)):
            self.node_to_index[self.nodes[i]] = i
        self.bias = bias
        
        self.adj_matrix = [[math.inf for i in range(self.num_nodes)] for j in range(self.num_nodes)]
        for start_node, end_node, edge_weight in edges:
            self.adj_matrix[self._node_to_index(start_node)][self._node_to_index(end_node)] = edge_weight

    #dijkstra's
    def _shortest_path(self, adj_matrix : list, start_node : str, goal_node : str):
        start = self._node_to_index(start_node)
        unvisited = [i for i in range(len(self.nodes))]
        visited = []
        shortest_distance = [math.inf for i in self.nodes]
        shortest_distance[start] = 0
        previous_vertex = [None for i in self.nodes]
        while True:
            outgoing_edges = adj_matrix[start]
            for i in range(len(outgoing_edges)):
                edge_weight = outgoing_edges[i]
                if i in unvisited and edge_weight != math.inf:
                    distance = shortest_distance[start] + edge_weight
                    if distance < shortest_distance[i]:
                        shortest_distance[i] = distance
                        previous_vertex[i] = start
            visited.append(start)
            unvisited.remove(start)
            if len(unvisited) == 0:
                break
            min = math.inf
            node = None
            for i in range(len(outgoing_edges)):
                if shortest_distance[i] < min and i in unvisited:
                    min = shortest_distance[i]
                    node = i
            start = node

        start = self._node_to_index(start_node)
        
        curr = self._node_to_index(goal_node)
        total_distance = 0
        path = []
        path.append(self.nodes[curr])
        while curr != start:
            prev = previous_vertex[curr]
            total_distance += adj_matrix[prev][curr]
            path.insert(0, self.nodes[prev])
            curr = prev
        return path, total_distance
            


    def _node_to_index(self, node):
        return self.node_to_index[node]

    #present, variable bias
    def _calc_variable_bias(self):
        return random.uniform(0, 1)

    def delete_edges(self):
        pass

    def traverse_optimal(self, start_node : str, goal_node : str):
        path, distance = self._shortest_path(self.adj_matrix, start_node, goal_node)
        return path, distance

    def traverse_procrastination(self, start : str, goal : str):
        pass


if __name__ == "__main__":
    nodes = ["a", "b", "c", "d", "e", "f", "g"]
    edges = [("a", "f", 16), ("a", "c", 8), ("f", "b", 2), ("c", "d", 8), ("d", "g", 8), ("c", "e", 2), ("e", "g", 16), ("b", "g", 2)]

    tg = TaskGraph(nodes, edges, "constant")

    tg._shortest_path(tg.adj_matrix, "a", "g")
    pass