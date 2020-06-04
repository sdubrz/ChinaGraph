import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors


# 使用 local PCA的降维方法
class LocalPCADR:
    def __init__(self, n_components=2, affinity='expCov', parameters={}, algorithm='MDS'):
        """
        初始化方法
        :param n_components: 降维之后的维度数
        :param affinity: 相似度的计算类型
                        'expCov': 同时考虑协方差矩阵的相似性和欧氏距离的相似性，并转化为指数形式
                        'expQ': 同时考虑投影的相似性与欧氏距离的相似性，并转化为指数形式
                        'cov': 同时考虑协方差矩阵的相似性与欧氏距离的相似性，它们两项的加权和
                        'Q': 同时考虑投影的相似性与欧氏距离的相似性，它们两项的加权和
        :param parameters: 一个参数字典，里面是欧氏距离与 local PCA的权重，下面是几个常用的参数
                        'alpha': 欧氏距离的权重
                        'beta': local PCA 的权重
                        'neighborhood_type': 计算邻居的方式
                                            'knn': K近邻方法
                                            'rnn': 设置邻域半径的方法
                        'n_neighbors': 邻域内点的个数，当 neighborhood_type == 'knn' 时有效
                        'neighborhood_size': 局部邻域的半径大小，当 neighborhood_type == 'rnn' 时有效
        :param algorithm: 迭代求解降维结果时用的框架
                            'MDS': 使用 SAMCOF 算法求解
                            't-SNE': 使用 t-SNE 的迭代方式求解
        """
        self.n_components = n_components
        self.affinity = affinity
        self.parameters = parameters

    def local_cov_knn(self, X, n_neighbors):
        """
        用 KNN 的方式计算 local covariance matrix
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :param n_neighbors: 邻域内邻居的个数
        :return:
        """
        (n, m) = X.shape
        M = np.zeros((n, m, m))

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distance, knn = nbrs.kneighbors(X)
        for i in range(0, n):
            local_data = np.zeros((n_neighbors, m))
            for j in range(0, n_neighbors):
                local_data[j, :] = X[knn[i, j], :]
            local_mean = np.mean(local_data, axis=0)
            local_data = local_data - local_mean
            Ci = np.matmul(local_data.T, local_data) / n_neighbors
            M[i, :, :] = Ci[:, :]

        return M

    def local_cov_rnn(self, X, neighborhood_size):
        """
        用 KNN 的方式计算 local matrix covariance matrix
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :param neighborhood_size: 邻域半径的大小
        :return:
        """
        (n, m) = X.shape
        M = np.zeros((n, m, m))
        # 尚未完成
        return M

    def local_cov(self, X):
        """
        计算每个点的 local covariance matrix
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return: 一个三维的张量，其中 M[i, :, :] 为第 i 个点的 local covariance matrix
        """
        (n, m) = X.shape
        M = np.zeros((n, m, m))

        if self.parameters['neighborhood_type'] == 'knn':
            print("Calculate local covariance matrix using KNN...")
            M = self.local_cov_knn(X, self.parameters['n_neighbors'])
        elif self.parameters['neighborhood_type'] == 'rnn':
            print("Calculate local covariance matrix using RNN...")
            M = self.local_cov_rnn(X, self.parameters['neighborhood_size'])
        else:
            print("Wrong neighborhood_type: "+str(self.parameters['neighborhood_type']))
            return

        return M

    def affinity_matrix(self, X):
        """
        计算 affinity 矩阵
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return:
        """
        (n, m) = X.shape
        W = np.zeros((n, n))
        D = euclidean_distances(X)  # 欧氏距离矩阵

        if self.affinity == 'cov':
            # 协方差矩阵距离与欧氏距离加权和
            neighborhood_size = self.parameters['neighborhood_size']

        return W

    def fit_transform(self, X):
        """
        计算降维结果
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return:
        """
        (n, m) = X.shape

