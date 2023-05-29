import random
import math
import sys
from copy import deepcopy

class TaskGraph:

    def __init__(self, nodes , edges ):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.node_to_index = {}
        for i in range(len(self.nodes)):
            self.node_to_index[self.nodes[i]] = i
        
        self.adj_matrix = [[math.inf for i in range(self.num_nodes)] for j in range(self.num_nodes)]
        for start_node, end_node, edge_weight in edges:
            self.adj_matrix[self._node_to_index(start_node)][self._node_to_index(end_node)] = edge_weight
        
        self.edges = edges

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
        
        # backtracking to get path and total_distance
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
        # bias is drawn iid from [1, inf) (2016, p. 5)
        return random.uniform(0, sys.maxsize)

    def delete_edges(self):
        pass

    def traverse_optimal(self, start_node : str, goal_node : str):
        path, distance = self._shortest_path(self.adj_matrix, start_node, goal_node)
        return path, distance
    
    def traverse_variable_procrastination(self, start_node : str, goal_node : str):
        curr_node = start_node

        total_distance = 0
        final_path = []
        final_path.append(curr_node)
        
        while curr_node != goal_node:
            
            adj_matrix = deepcopy(self.adj_matrix)
            curr = self._node_to_index(curr_node)
            
            bias = self._calc_variable_bias()
            self._scale_adj_matrix_variable(adj_matrix, curr, bias)
            path, _ = self._shortest_path(adj_matrix, curr_node, goal_node)
            next_node = path[1]
            next = self._node_to_index(next_node)
            distance = self.adj_matrix[curr][next]
            del adj_matrix
            total_distance += distance
            curr_node = next_node
            final_path.append(curr_node)

        return final_path, total_distance

    def traverse_constant_procrastination(self, start_node : str, goal_node : str, bias: float=None, reward: float=None):
        curr_node = start_node
        try:
            scaled_reward = bias * reward # reward is always perceived as scaled (2016, page. 6)
        except Exception as e:
            pass

        total_distance = 0
        final_path = []
        final_path.append(curr_node)
        
        while curr_node != goal_node:
            
            adj_matrix = deepcopy(self.adj_matrix)
            curr = self._node_to_index(curr_node)
            
            self._scale_adj_matrix_constant(adj_matrix, curr, bias)
            path, preceived_distance = self._shortest_path(adj_matrix, curr_node, goal_node)
            next_node = path[1]
            next = self._node_to_index(next_node)
            distance = self.adj_matrix[curr][next]
            del adj_matrix

            # please double check the preceived_distance > scaled_reward, as I thought it included what had been previously travelled but that returns the wrong behavior for the projects example problem 
            if reward is not None and preceived_distance > scaled_reward: # abandonment
                break
            total_distance += distance
            curr_node = next_node
            final_path.append(curr_node)

        return final_path, total_distance
    
    def _scale_adj_matrix_constant(self, adj_matrix, curr, bias):
        for i in range(len(adj_matrix)):
            if i != curr:
                adj_matrix[i] = [x * bias for x in adj_matrix[i]]
            

    def _scale_adj_matrix_variable(self, adj_matrix, curr, bias):
        # scales weights so b * immediate edges, leaving remaining unchanged (2016, p. 5)
        adj_matrix[curr] = [x * bias for x in adj_matrix[curr]]

    
    def _plot_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()

        for edge in self.edges:
            G.add_edge(edge[0],edge[1],weight=edge[2])
        
        elarge = [(u, v) for (u, v, d) in G.edges(data=True)]

        pos = nx.kamada_kawai_layout(G)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3)
        

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        

class TwoStateTaskGraph:
    def __init__(self, num_days , do_cost, procrastinate_cost ):
        self.adj_list = self._create_two_state_task_graph(num_days, do_cost=2, procrastinate_cost=1)


    def _create_two_state_task_graph(self, num_days, do_cost=2, procrastinate_cost=1):
        '''
            This method creates a two state task graph. Two states are done and undone. Ref 2016 paper
            num_days is the total number of days the task needs to be done
            This simulates task graph of bounded and monotone distance property
        '''
        n=num_days
        adj_list = {}
        
        for i in range(0,n):
            if i==0:
                adj_list['S'] = [('D'+str(i+1),do_cost), ('P'+str(i+1),procrastinate_cost)]
            else:
                adj_list['D'+str(i)] = [('D'+str(i+1),0)]
                if i+1 != n:
                    adj_list['P'+str(i)] = [('D'+str(i+1),do_cost), ('P'+str(i+1),procrastinate_cost)]
                else:
                    adj_list['P'+str(i)] = [('D'+str(i+1),do_cost)]

        adj_list['D'+str(n)] = []

        return adj_list

    
    def _plot_two_state_task_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        pos={}
        p_idx = 0.5
        d_idx = 0.5
        for node in self.adj_list:
            if node[0] == 'S':
                pos[node]=(0,0.5)
            elif node[0] == 'P':
                pos[node]=(p_idx,0)
                p_idx+=0.5
            else:
                pos[node]=(d_idx,1)
                d_idx+=0.5


        # Create a new graph
        G = nx.Graph()

        # Add edges and weights to the graph
        for node, neighbors in self.adj_list.items():
            for neighbor, weight in neighbors:
                G.add_edge(node, neighbor, weight=weight)

        # Draw the graph
        nx.draw(G, pos, with_labels=True)  # draw nodes
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))  # draw edge labels
        plt.show()  