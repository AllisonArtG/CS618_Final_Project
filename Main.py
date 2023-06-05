import random

from TaskGraph import TaskGraph
from TaskGraph import TwoStateTaskGraph
import pprint
import matplotlib.pyplot as plt
import numpy as np

def main():
    random.seed(42)

    nodes = ["a", "b", "c", "d", "e", "f", "g"]
    edges = [("a", "f", 16), ("a", "c", 8), ("f", "b", 2), ("c", "d", 8), ("d", "g", 8), ("c", "e", 2), ("e", "g", 16), ("b", "g", 2)]

    tg = TaskGraph(nodes, edges)

    tg._plot_graph()

    print("\nSimple Graph Example\n")

    print("Optimal Shortest Path")
    path, distance = tg._shortest_path(tg.adj_matrix, "a", "g")
    print("path:", path)
    print("distance", distance, "\n")

    constant_bias = 0.5 # referred to as Beta in 2014 paper

    print("Constant Bias Procrastination")
    print("bias:", constant_bias)
    path, distance = tg.traverse_constant_procrastination("a", "g", constant_bias)
    print("path:", path)
    print("distance", distance, "\n")

    print("Constant Bias Procrastination w/ End Reward")
    print("Demostrates Task Abandonment")
    print("bias:", constant_bias)
    reward = 31
    print("end reward:", reward)
    path, distance = tg.traverse_constant_procrastination("a", "g", constant_bias, reward)
    print("path:", path)
    print("distance", distance, "\n")

    print("Variable Bias Procrastination")
    path, distance = tg.traverse_variable_procrastination("a", "g")
    print("path:", path)
    print("distance", distance, "\n")

    print("Course (and Two Projects) Example\n")

    nodes = ["s", "v_10", "v_20", "v_30", "v_11", "v_21", "v_31", "v_12", "v_22", "g"]
    edges = [("s", "v_10", 1), ("v_10", "v_20", 1), ("v_20", "v_30", 1), 
              ("v_11", "v_21", 1), ("v_21", "v_31", 1), 
              ("v_12", "v_22", 1), ("v_22", "g", 1),
              ("s", "v_11", 4), ("v_10", "v_21", 4), ("v_20", "v_31", 4),
              ("v_11", "v_22", 4), ("v_21", "g", 4),
              ("s", "v_12", 9), ("v_10", "v_22", 9), ("v_20", "g", 9)]
    reward = [("g", 16)]

    tg_student_ex = TaskGraph(nodes, edges)
    tg_student_ex._plot_graph()

    path, distance = tg_student_ex.traverse_optimal("s", "g")
    print("path:", path)
    print("distance", distance, "\n")

    # print("Constant Bias Procrastination w/ End Reward")
    # print("Demostrates Task Abandonment")
    # print("bias:", constant_bias)
    # print("reward:", reward)
    path, distance = tg_student_ex.traverse_constant_procrastination("s", "g", constant_bias, reward[0][1])
    print("path:", path)
    print("distance", distance, "\n")

    nodes.remove("v_20")
    nodes.remove("v_30")
    edges.remove(("v_10", "v_20", 1)) 
    edges.remove(("v_20", "v_30", 1))
    edges.remove(("v_20", "v_31", 4))
    edges.remove(("v_20", "g", 9))

    tg_student_ex_trimmed = TaskGraph(nodes, edges)
    path, distance = tg_student_ex_trimmed.traverse_constant_procrastination("s", "g", constant_bias, reward[0][1])
    print("path:", path)
    print("distance", distance, "\n")


    print("Simulations for 2 states Task Graphs\n")

    twoStateTaskGraph = TwoStateTaskGraph(6, 2, 1)
    twoStateTaskGraph._plot_two_state_task_graph()

    print("Task Graph in Ajacency List: \n")
    pprint.PrettyPrinter(width=20, sort_dicts=False).pprint(twoStateTaskGraph._get_adj_list())

    path_without_bias = twoStateTaskGraph._traverse_with_constant_bias(bias_factor = 0)
    path_with_constant_bias = twoStateTaskGraph._traverse_with_constant_bias(bias_factor = 2)

    path_with_variable_bias = twoStateTaskGraph._traverse_with_variable_bias()
   
    print("\nTraversal of agent without present bias", path_without_bias[0],"\nCost:", path_without_bias[1])
    print("\nTraversal of agent with constant present bias",path_with_constant_bias[0],"\nCost:", path_with_constant_bias[1])
    print("\nTraversal of agent with variable present bias",path_with_variable_bias[0],"\nCost:", path_with_variable_bias[1])

    print("\n Now simulating the task graphs for different number of days and with 3 conditions: No bias, constant bias and variable bias\n")
    
    # Add sizes to this list to simulate different days
    graph_sizes = [6,10,25,30,45,60,100]

    cost_without_bias_list, cost_with_constant_bias_list, cost_with_variable_bias_list = simulate_graph_traversal_diff_bias(graph_sizes)

    plot_cost_graph([graph_sizes, cost_without_bias_list, cost_with_constant_bias_list, cost_with_variable_bias_list])

    # Simulate graph traversal cost for different distribution of present bias selection
    pdf_list = [[1/3, 2/3],[1/6, 5/6],[1/12, 11/12],[1/24,23/24]]
    cost_with_variable_bias_all_pdf_list = simulate_graph_traversal_pdf_bias(graph_sizes, pdf_list)

    plot_cost_graph_different_pdf(pdf_list, graph_sizes, cost_with_variable_bias_all_pdf_list)

def simulate_graph_traversal_diff_bias(graph_sizes):
    
    cost_without_bias_list = []
    cost_with_constant_bias_list = []
    cost_with_variable_bias_list = []

    for graph_size in graph_sizes:
        print("\n Simulating graph traversal for task graph with num_days: ", graph_size)
        twoStateTaskGraph = TwoStateTaskGraph(graph_size, 2, 1)

        path_without_bias = twoStateTaskGraph._traverse_with_constant_bias(bias_factor = 0)
        path_with_constant_bias = twoStateTaskGraph._traverse_with_constant_bias(bias_factor = 2)


        cost_without_bias = path_without_bias[1]
        cost_with_constant_bias = path_with_constant_bias[1]

        total_cost_variable_bias = 0
        iterations = 1000
        for i in range (iterations):
            path_with_variable_bias = twoStateTaskGraph._traverse_with_variable_bias()
            total_cost_variable_bias+=path_with_variable_bias[1]

        avg_cost_variable_bias = total_cost_variable_bias/iterations

        cost_without_bias_list.append(cost_without_bias)
        cost_with_constant_bias_list.append(cost_with_constant_bias)
        cost_with_variable_bias_list.append(avg_cost_variable_bias)

    return cost_without_bias_list, cost_with_constant_bias_list, cost_with_variable_bias_list


def simulate_graph_traversal_pdf_bias(graph_sizes, pdf_list) :
    # Now to show that cost of traversal only depends on the distribution of bias in variable present bias setting
    cost_with_variable_bias_all_pdf_list = []

    for graph_size in graph_sizes:
        print("\n Simulating graph traversal for task graph with num_days: ", graph_size)

        twoStateTaskGraph = TwoStateTaskGraph(graph_size, 2, 1)

        total_cost_variable_bias = 0
        iterations = 1000

        cost_with_variable_bias_list  = []
        for probabilities in pdf_list:
            
            for i in range (iterations):
                path_with_variable_bias = twoStateTaskGraph._traverse_with_variable_bias(probabilities=probabilities)
                total_cost_variable_bias+=path_with_variable_bias[1]

            avg_cost_variable_bias = total_cost_variable_bias/iterations
            cost_with_variable_bias_list.append(avg_cost_variable_bias)

        cost_with_variable_bias_all_pdf_list.append(cost_with_variable_bias_list)

    cost_with_variable_bias_all_pdf_list = np.array(cost_with_variable_bias_all_pdf_list).T
    return cost_with_variable_bias_all_pdf_list

def plot_cost_graph(array):
    graph_sizes = array[0]
    cost_without_bias_list = array[1]
    cost_with_constant_bias_list = array[2]
    cost_with_variable_bias_list = array[3]

    plt.plot(graph_sizes, cost_without_bias_list, label="Cost w/o present bias")
    plt.plot(graph_sizes, cost_with_constant_bias_list, label='Cost with constant bias ')
    plt.plot(graph_sizes, cost_with_variable_bias_list, label='Cost with variable bias')

    plt.xlabel('Deadline days from start')
    plt.ylabel('Cost')
    plt.title('Traversal cost of task graph for different bias parameters')
    plt.legend()
    plt.show()


def plot_cost_graph_different_pdf(pdf_list, graph_sizes, cost_with_variable_bias_all_pdf_list):
    for i in range(len(pdf_list)):
        plt.plot(graph_sizes, cost_with_variable_bias_all_pdf_list[i], label=str([round(num, 2) for num in pdf_list[i]]))
    plt.xlabel('Deadline days from start')
    plt.ylabel('Cost')
    plt.title('Traversal cost of task graph for different distribution for bias factor [1,3]')
    plt.legend(loc='upper right')
    plt.show()



if __name__ == "__main__":
    main()

