import random

from TaskGraph import TaskGraph

def main():
    random.seed(42)

    nodes = ["a", "b", "c", "d", "e", "f", "g"]
    edges = [("a", "f", 16), ("a", "c", 8), ("f", "b", 2), ("c", "d", 8), ("d", "g", 8), ("c", "e", 2), ("e", "g", 16), ("b", "g", 2)]

    tg = TaskGraph(nodes, edges)

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

    # student project example
    nodes = ["s", "v_10", "v_20", "v_30", "v_11", "v_21", "v_31", "v_12", "v_22", "g"]
    edges = [("s", "v_10", 1), ("v_10", "v_20", 1), ("v_20", "v_30", 1), 
              ("v_11", "v_21", 1), ("v_21", "v_31", 1), 
              ("v_12", "v_22", 1), ("v_22", "g", 1),
              ("s", "v_11", 4), ("v_10", "v_21", 4), ("v_20", "v_31", 4),
              ("v_11", "v_22", 4), ("v_21", "g", 4),
              ("s", "v_12", 9), ("v_10", "v_22", 9), ("v_20", "g", 9)]
    reward = [("g", 16)]

    tg_student_ex = TaskGraph(nodes, edges)

    path, distance = tg_student_ex.traverse_optimal("s", "g")
    print("path:", path)
    print("distance", distance, "\n")

    # print("Constant Bias Procrastination w/ End Reward")
    # print("Demostrates Task Abandonment")
    # print("bias:", constant_bias)
    # print("rewards:", rewards)
    path, distance = tg_student_ex.traverse_procrastination("s", "g", "constant", constant_bias, reward[0][1])
    print("path:", path)
    print("distance", distance, "\n")

    nodes.remove("v_20")
    nodes.remove("v_30")
    edges.remove(("v_10", "v_20", 1)) 
    edges.remove(("v_20", "v_30", 1))
    edges.remove(("v_20", "v_31", 4))
    edges.remove(("v_20", "g", 9))

    tg_student_ex_trimmed = TaskGraph(nodes, edges)
    path, distance = tg_student_ex_trimmed.traverse_procrastination("s", "g", "constant", constant_bias, reward[0][1])
    print("path:", path)
    print("distance", distance, "\n")





if __name__ == "__main__":
    main()

