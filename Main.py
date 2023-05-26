import random

from TaskGraph import TaskGraph

def main():
    random.seed(42)

    nodes = ["a", "b", "c", "d", "e", "f", "g"]
    edges = [("a", "f", 16), ("a", "c", 8), ("f", "b", 2), ("c", "d", 8), ("d", "g", 8), ("c", "e", 2), ("e", "g", 16), ("b", "g", 2)]

    tg = TaskGraph(nodes, edges, "constant")

    # print("Optimal Shortest Path")
    # path, distance = tg._shortest_path(tg.adj_matrix, "a", "g")
    # print("path:", path)
    # print("distance", distance, "\n")

    constant_bias = 1/0.5 #Beta is 0.5 and b is inverse of Beta

    # print("Constant Bias Procrastination")
    # print("bias:", constant_bias)
    # path, distance = tg.traverse_procrastination("a", "g", "constant", constant_bias)
    # print("path:", path)
    # print("distance", distance, "\n")

    # print("Constant Bias Procrastination w/ End Reward")
    # print("Demostrates Task Abandonment")
    # print("bias:", constant_bias)
    # reward = 35
    # print("end reward:", reward)
    # path, distance = tg.traverse_procrastination("a", "g", "constant", constant_bias, reward)
    # print("path:", path)
    # print("distance", distance, "\n")

    # print("Variable Bias Procrastination")
    # path, distance = tg.traverse_procrastination("a", "g", "variable")
    # print("path:", path)
    # print("distance", distance, "\n")

if __name__ == "__main__":
    main()

