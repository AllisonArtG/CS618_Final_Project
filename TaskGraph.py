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
        goal = self._node_to_index(goal_node)
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
            if start == goal or len(unvisited) == 0:
                break
            min = math.inf
            node = None
            for i in range(len(outgoing_edges)):
                if shortest_distance[i] < min and i in unvisited:
                    min = shortest_distance[i]
                    node = i
            start = node

        start = self._node_to_index(start_node)
        
        curr = goal
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
        return 1/random.uniform(0, 1)

    def delete_edges(self):
        pass

    def traverse_optimal(self, start_node : str, goal_node : str):
        path, distance = self._shortest_path(self.adj_matrix, start_node, goal_node)
        return path, distance

    def traverse_procrastination(self, start_node : str, goal_node : str, bias_type: str, bias: float=None, reward: float=None):
        curr_node = start_node

        total_distance = 0
        total_prec_distance = 0
        final_path = []
        final_path.append(curr_node)
        
        while curr_node != goal_node:
            
            adj_matrix = deepcopy(self.adj_matrix)
            curr = self._node_to_index(curr_node)
            
            if bias_type == "variable":
                bias = self._calc_variable_bias()
            self._scale_adj_matrix(adj_matrix, curr, bias)
            path, preceived_distance = self._shortest_path(adj_matrix, curr_node, goal_node)
            next_node = path[1]
            next = self._node_to_index(next_node)
            distance = self.adj_matrix[curr][next]
            prec_distance = adj_matrix[curr][next]
            del adj_matrix
            if reward is not None and total_prec_distance + preceived_distance > reward: # abandonment
                break
            total_distance += distance
            total_prec_distance += prec_distance
            curr_node = next_node
            final_path.append(curr_node)

        return final_path, total_distance
            

    def _scale_adj_matrix(self, adj_matrix, curr, bias):
        adjacent = adj_matrix[curr]
        scaled_adj = []
        for edge_weight in adjacent:
            if edge_weight != math.inf:
                new_weight = edge_weight * bias
                scaled_adj.append(new_weight)
            else:
                scaled_adj.append(edge_weight)
        adj_matrix[curr] = scaled_adj
        
    




if __name__ == "__main__":
    random.seed(42)

    nodes = ["a", "b", "c", "d", "e", "f", "g"]
    edges = [("a", "f", 16), ("a", "c", 8), ("f", "b", 2), ("c", "d", 8), ("d", "g", 8), ("c", "e", 2), ("e", "g", 16), ("b", "g", 2)]

    tg = TaskGraph(nodes, edges, "constant")

    #path, distance = tg._shortest_path(tg.adj_matrix, "a", "g")

    constant_bias = 1/0.5 #Beta is 0.5 and b is inverse of Beta

    #path, distance = tg.traverse_procrastination("a", "g", "constant", constant_bias)

    # demostrates abandonment at node c
    path, distance = tg.traverse_procrastination("a", "g", "constant", constant_bias, 35)

    #path, distance = tg.traverse_procrastination("a", "g", "variable")