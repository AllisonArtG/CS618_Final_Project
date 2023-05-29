import random

from TaskGraph import TwoStateTaskGraph
import pprint
import matplotlib.pyplot as plt
import numpy as np

def main():
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