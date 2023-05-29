import random

from TaskGraph import TwoStateTaskGraph
import pprint

def main():
    twoStateTaskGraph = TwoStateTaskGraph(6, 2, 1)
    #twoStateTaskGraph._plot_two_state_task_graph()

    print("Task Graph in Ajacency List: \n")
    pprint.PrettyPrinter(width=20, sort_dicts=False).pprint(twoStateTaskGraph._get_adj_list())

    path_without_bias = twoStateTaskGraph._traverse_with_constant_bias('D6',bias_factor = 0)
    path_with_constant_bias = twoStateTaskGraph._traverse_with_constant_bias('D6',bias_factor = 2)

    path_with_variable_bias = twoStateTaskGraph._traverse_with_variable_bias('D6')
   
    print("\nTraversal of agent without present bias", path_without_bias[0],"\nCost:", path_without_bias[1])
    print("\nTraversal of agent with constant present bias",path_with_constant_bias[0],"\nCost:", path_with_constant_bias[1])
    print("\nTraversal of agent with variable present bias",path_with_variable_bias[0],"\nCost:", path_with_variable_bias[1])

if __name__ == "__main__":
    main()