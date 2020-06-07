import numpy as np
from sklearn.utils.graph import graph_shortest_path
# 测试一些图算法的使用


def shortest_path():
    A = np.array([[0, 5, 0, 7],
                  [0, 0, 4, 2],
                  [3, 3, 0, 2],
                  [0, 0, 1, 0]])

    distance = graph_shortest_path(A, directed=True)
    print(distance)


if __name__ == '__main__':
    shortest_path()

