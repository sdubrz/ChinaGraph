import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors


def mahalanobis_distance(A, B):
    """
    计算两个矩阵之间的马氏距离
    这里用的是矩阵差的最大特征值
    :param A:
    :param B:
    :return:
    """
    C = A - B
    values, vectors = np.linalg.eig(C)
    values = np.abs(values)
    return np.max(values)


def cov_distance(X, n_neighbors):
    """
    计算每个点的协方差矩阵之间的距离
    :param X: 样本矩阵，每一行是一个样本
    :param neighbors:
    :return:
    """
    print("Mahalanobis distance of covariance matrixs...")
    (n, m) = X.shape
    nbr_s = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distance, KNN = nbr_s.kneighbors(X)

    # 计算每个点的 local 协方差矩阵
    COV = np.zeros((n, m, m))
    for i in range(0, n):
        local_data = np.zeros((n_neighbors, m))
        for j in range(0, n_neighbors):
            local_data[j, :] = X[KNN[i, j], :]

        mean_x = np.mean(local_data, axis=0)
        local_data = local_data - mean_x
        cov = np.dot(local_data.T, local_data) / n_neighbors
        COV[i] = cov

    D = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):  # 先这么写看看对称情况
            if i!=j:
                D[i, j] = mahalanobis_distance(COV[i], COV[j])

    return D


def Q_distance(X, n_neighbors):
    """
    计算每个点的local PCA中的特征向量组成的矩阵之间的距离
    这个手工推导还有点问题
    :param X: 数据矩阵，每一行是一个点
    :param n_neighbors: 近邻数
    :return:
    """
    print("Mahalanobis distance of Q matrixs...")
    (n, m) = X.shape
    nbr_s = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distance, KNN = nbr_s.kneighbors(X)

    # 计算每个点的 local 协方差矩阵
    COV = np.zeros((n, m, m))
    for i in range(0, n):
        local_data = np.zeros((n_neighbors, m))
        for j in range(0, n_neighbors):
            local_data[j, :] = X[KNN[i, j], :]

        mean_x = np.mean(local_data, axis=0)
        local_data = local_data - mean_x
        cov = np.dot(local_data.T, local_data) / n_neighbors
        COV[i] = cov

    D = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):  # 先这么写看看对称情况
            if i != j:
                D[i, j] = mahalanobis_distance(COV[i], COV[j])

    return D


def distance_and_cov(X, n_neighbors, alpha):
    """
    距离与协方差矩阵相似性相加
    :param X: 数据矩阵，每一行是一个样本
    :param n_neighbors: 邻居数，是包括数据点本身的
    :param alpha: 协方差的参数
    :return:
    """
    (n, m) = X.shape
    print("euclidean distance...")
    DE = euclidean_distances(X)  # 欧氏距离矩阵
    DC = cov_distance(X, n_neighbors)  # 协方差矩阵的距离

    DE = DE / np.max(DE)
    DC = DC / np.max(DC)
    D = DE*(1-alpha) + DC*alpha

    return D


def distance_multi_cov(X, n_neighbors):
    """
    距离与协方差矩阵相乘
    :param X:
    :param n_neighbors:
    :return:
    """
    (n, m) = X.shape
    print("euclidean distance...")
    DE = euclidean_distances(X)  # 欧氏距离矩阵
    DC = cov_distance(X, n_neighbors)  # 协方差矩阵的距离

    D = DE * DC

    return D







