import random

from TaskGraph import TwoStateTaskGraph

def main():
    twoStateTaskGraph = TwoStateTaskGraph(6, 2, 1)
    twoStateTaskGraph._plot_two_state_task_graph()

if __name__ == "__main__":
    main()