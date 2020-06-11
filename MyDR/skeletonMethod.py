import numpy as np
from sklearn.neighbors import NearestNeighbors
import random


# 用骨架点的方法降维所需的一些函数


def get_skeleton(X, neighborhood_type='knn', n_neighbors=8, neighborhood_size=0.1, label=None):
    """
    划分骨架
    会出现一个点属于多个点的势力范围的情况，所以这里不用记录非骨架点的归宿，后面再用一个 k 近邻方法来算就可以了
    :param X: 高维数据矩阵，每一行是一个高维数据点
    :param neighborhood_type: 求邻居的方法，可取 'knn' 或 'rnn'
    :param n_neighbors: 当 neighborhood_type == 'knn' 时，邻居的个数
    :param neighborhood_size: 当 neighborhood_type == 'rnn' 时，邻域的半径大小
    :param label: 所有数据的标签
    :return: skeleton 骨架点的索引
            satellite 每个骨架点周围的点
            ##### skeleton_label 记录骨架点标签的 list，当 label is None 时，直接返回 None
            # 当前的数据结构来看，这个骨架点的 label 不好筛选出来
    """
    (n, d) = X.shape
    skeleton = []  # 骨架点的索引
    satellite = []  # 记录每个骨架点周围都有哪些点
    skeleton_label = None  # 记录骨架点的标签
    # if not (label is None):
    #     skeleton_label = []
    # selected = np.zeros((n, 1))  # 记录该点是否被选中过

    neighbor_lists = []
    if neighborhood_type == 'knn':
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distance, knn = nbrs.kneighbors(X)
        neighbor_lists = knn.tolist()
    else:
        neigh = NearestNeighbors(radius=neighborhood_size).fit(X)
        distance, rnn = neigh.radius_neighbors(X)
        neighbor_lists = rnn

    rest = {x for x in range(n)}  # 剩余的没有选中的点
    rest, total_label = random_disruption(rest, label)  # 随机打散
    while len(rest) > 0:
        current = rest.pop()  # 这种方法貌似不是随机的
        neighbors = neighbor_lists[current]
        skeleton.append(current)
        satellite.append(neighbors)
        for p in neighbors:
            if p in rest:
                rest.remove(p)

    return skeleton, satellite


def random_disruption(index_set0, label0=None):
    """
    随机打乱顺序
    :param index_set0: 所有数据的序号
    :param label0: 所有数据的标签 np.array((n, 1))
    :return:
    """
    import copy
    n = len(index_set0)
    index_set = copy.copy(index_set0)
    label = copy.copy(label0)
    for i in range(n):
        a = random.randint(0, n-1)
        if a == n-1:
            continue
        index_set.remove(a)
        index_set.add(a)
        if not (label is None):
            a_label = label[a]
            for j in range(a, n-1):
                label[j] = label[j+1]
            label[n-1] = a_label

    return index_set, label


def get_skeleton_cov(X, satellite):
    """
    计算每个点的 local covariance matrix
    :param X: 原始的高维数据矩阵，每一行是一个高维数据点
    :param satellite: 每个骨架点的卫星数据
    :return:
    """
    (n, d) = X.shape
    n2 = len(satellite)
    Cov = np.zeros((n2, d, d))

    index = 0
    for neighbors in satellite:
        local_data = np.zeros((len(neighbors), d))
        count = 0
        for x in neighbors:
            local_data[count, :] = X[x, :]
            count = count + 1
        local_data = local_data - np.mean(local_data, axis=0)
        if len(neighbors) > 1:
            C = np.matmul(local_data.T, local_data) / len(neighbors)
        else:
            C = np.outer(local_data.T, local_data)
        Cov[index, :, :] = C[:, :]
        index = index + 1
    return Cov


def satellite_location(X, skeleton, skeleton_location):
    """
    根据骨架点的降维结果确定所有点（主要是非骨架点）的降维坐标
    :param X: 高维数据矩阵，每一行是一个样本点
    :param skeleton: 骨架点的索引号
    :param skeleton_location: 骨架点的降维坐标
    :return:
    """
    (n, d) = X.shape
    (n2, manifold_dimension) = skeleton_location.shape
    Y = np.zeros((n, manifold_dimension))

    skeleton_X = np.zeros((n2, d))
    for i in range(0, n2):
        skeleton_X[i, :] = X[skeleton[i], :]
    nbrs = NearestNeighbors(n_neighbors=manifold_dimension+1, algorithm='ball_tree').fit(skeleton_X)
    distance, knn = nbrs.kneighbors(X)
    for i in range(0, n):
        if i in skeleton:
            Y[i, :] = skeleton_location[skeleton.index(i), :]
        else:
            anchors = []  # 锚点，即用于确定该点坐标的骨架点
            for j in range(0, manifold_dimension+1):
                anchors.append(skeleton[knn[i, j]])

            A = np.zeros((manifold_dimension+1, manifold_dimension+1))
            B = np.ones((manifold_dimension+1, 1))
            for j in range(0, manifold_dimension+1):
                A[j, 0:manifold_dimension] = X[anchors[j], 0:manifold_dimension]
            A[:, manifold_dimension] = 1
            A = A.T
            for j in range(0, manifold_dimension):
                B[j] = X[i, j]
            w = np.linalg.solve(A, B)
            for j in range(0, manifold_dimension+1):
                Y[i, :] = Y[i, :] + w[j] * skeleton_location[knn[i, j], :]

    return Y


def selected_finished(selected):
    """
    检查是否每个点都被筛选过了
    :param selected:
    :return:
    """
    for i in selected:
        if i == 0:
            return False
    return True


def solve_test():
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 1, 1]])
    B = np.array([[1], [1], [1]])
    C = np.linalg.solve(A, B)
    print(C)
    D = np.ones((4, 1))
    for i in range(0, 3):
        D[i] = A[0, i]
    print(D)
    A[2, :] = 0
    print(A)


if __name__ == '__main__':
    # test = {x for x in range(10)}
    # print(test)
    # test.remove(5)
    # print(test)
    # A = np.array([[1, 2], [3, 4]])
    # print(A.tolist())
    solve_test()
