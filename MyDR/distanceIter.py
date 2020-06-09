import numpy as np
from sklearn.neighbors import NearestNeighbors
# 用迭代的方法计算距离矩阵


def neighbors_by_distance(D, n_neighbors=10):
    """
    根据距离矩阵计算邻居
    :param D: 距离矩阵
    :param n_neighbors: 邻居个数
    :return:
    """
    (n, n1) = D.shape
    nbrs = NearestNeighbors(n_neighbors=n_neighbors-1, metric='precomputed').fit(D)
    distance, knn = nbrs.kneighbors()
    knn2 = np.zeros((n, n_neighbors))
    knn2[:, 1:n_neighbors] = knn[:, :]
    for i in range(0, n):
        knn2[i, 0] = i
    return knn2


def matrix_equals(A, B):
    """
    判断两个矩阵的值是否相等
    :param A:
    :param B:
    :return:
    """
    (n1, m1) = A.shape
    (n2, m2) = B.shape
    if n1 != n2 or m1 != m2:
        return False
    for i in range(0, n1):
        for j in range(0, m1):
            if A[i, j] != B[i, j]:
                return False
    return True


def knn_equals(A, B):
    """
    判断两个KNN是否相同
    :param A:
    :param B:
    :return:
    """
    (n, k) = A.shape
    for i in range(0, n):
        i_bool = True
        for j in range(0, k):
            j_bool = False
            for index in range(0, k):
                if int(A[i, j]) == int(B[i, index]):
                    j_bool = True
                    break
            if not j_bool:
                i_bool = False
                return False
        if not i_bool:
            return False
    return True


def neighbor_test():
    from sklearn.metrics import euclidean_distances
    A = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    D = euclidean_distances(A)
    nbrs = NearestNeighbors(n_neighbors=2, metric='precomputed').fit(D)
    distance, knn = nbrs.kneighbors()
    # 这里返回的KNN是不包括它自己的。
    print(distance)
    print(knn)


if __name__ == '__main__':
    neighbor_test()

