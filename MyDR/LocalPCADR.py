import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from MyDR import tsneFrame
from MyDR.geoTsne import geoTsne
from MyDR import distanceIter
from MyDR import skeletonMethod
from MyDR import tsneFramePlus


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


def local_cov_knn(X, n_neighbors):
    """
    用 KNN 的方式计算 local covariance matrix
    :param X: 高维数据矩阵，每一行是一个高维数据点
    :param n_neighbors: 邻域内邻居的个数
    :return:
    """
    (n, m) = X.shape
    print(X.shape)
    M = np.zeros((n, m, m))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distance, knn = nbrs.kneighbors(X)
    M = local_cov_by_knn(X, knn)

    return M


def local_cov_by_knn(X, knn):
    """
    在已知 KNN的情况下计算，local covariance matrix
    :param X: 高维数据矩阵，每一行是一个点
    :param knn: K近邻关系矩阵
    :return:
    """
    (n, m) = X.shape
    M = np.zeros((n, m, m))
    (n2, n_neighbors) = knn.shape

    for i in range(0, n):
        local_data = np.zeros((n_neighbors, m))
        for j in range(0, n_neighbors):
            local_data[j, :] = X[int(knn[i, j]), :]
        local_mean = np.mean(local_data, axis=0)
        local_data = local_data - local_mean
        Ci = np.matmul(local_data.T, local_data) / n_neighbors
        M[i, :, :] = Ci[:, :]

    return M


def local_cov_rnn(X, neighborhood_size):
    """
    用 KNN 的方式计算 local matrix covariance matrix
    :param X: 高维数据矩阵，每一行是一个高维数据点
    :param neighborhood_size: 邻域半径的大小
    :return:
    """
    (n, m) = X.shape
    M = np.zeros((n, m, m))

    neigh = NearestNeighbors(radius=neighborhood_size).fit(X)
    distance, rnn = neigh.radius_neighbors(X)
    index = 0
    small_count = 0

    for points in rnn:
        local_n = len(points)
        if local_n < 2:  # 对于该点，当前半径过小
            index += 1
            small_count += 1
            continue

        local_data = np.zeros((local_n, m))
        ptr = 0
        for j in points:
            local_data[ptr, :] = X[j, :]
            ptr += 1

        local_mean = np.mean(local_data, axis=0)
        local_data = local_data - local_mean
        Ci = np.matmul(local_data.T, local_data)
        M[index, :, :] = Ci[:, :]
        index += 1

    return M


def cov_matrix_distance(Cov, distance_type='spectralNorm'):
    """
    计算每个点的协方差矩阵之间的谱距离
    两个矩阵之间的谱距离，指的是这两个矩阵的差的谱范数
    :param Cov: 一个张量，存放的是每个点的 local covariance matrix
    :param distance_type: 协方差矩阵的距离类型
                        'spectralNorm': 谱范数，直接使用 numpy.linalg.norm(Matrix)
                        'mahalanobis': 马氏距离，对差矩阵进行特征值分解，取最大的特征值（假设所有特征值均为非负数）
    :return: 一个方阵，D[i, j] 为第 i 个点的协方差矩阵与第 j 个点的协方差矩阵之间的谱距离
    """
    (n, m, k) = Cov.shape
    D = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i + 1, n):
            if distance_type == 'spectralNorm':
                d = np.linalg.norm(Cov[i, :, :] - Cov[j, :, :])
            elif distance_type == 'mahalanobis':
                d = mahalanobis_distance(Cov[i, :, :], Cov[j, :, :])
            else:
                print("Wrong distance type of covariance matrix!")
                return
            D[i, j] = d
            D[j, i] = d
    return D


# 使用 local PCA的降维方法
class LocalPCADR:
    def __init__(self, n_components=2, affinity='expCov', parameters={}, frame='MDS'):
        """
        初始化方法
        :param n_components: 降维之后的维度数
        :param affinity: 相似度的计算类型
                        'expCov': 同时考虑协方差矩阵的相似性和欧氏距离的相似性，并转化为指数形式
                        'expQ': 同时考虑投影的相似性与欧氏距离的相似性，并转化为指数形式
                        'cov': 同时考虑协方差矩阵的相似性与欧氏距离的相似性，它们两项的加权和
                        'Q': 同时考虑投影的相似性与欧氏距离的相似性，它们两项的加权和
                        'MDS': 直接返回 MDS 的降维结果
                        't-SNE': 直接返回 t-SNE 的降维结果
                        'PCA': 直接返回 PCA的降维结果
                        'Isomap': 直接返回 Isomap 的降维结果
                        'LLE': 直接返回 LLE 的降维结果
                        'geo-t-SNE': 基于测地线距离的 t-SNE 方法
        :param parameters: 一个参数字典，里面是欧氏距离与 local PCA的权重，下面是几个常用的参数
                        'alpha': 欧氏距离的权重
                        'beta': local PCA 的权重
                        'neighborhood_type': 计算邻居的方式
                                            'knn': K近邻方法
                                            'rnn': 设置邻域半径的方法
                                            'iter': 用迭代的方式确定邻居，计算距离矩阵
                        'n_neighbors': 邻域内点的个数，当 neighborhood_type == 'knn' 时有效
                                        或当 affinity == 'Isomap' || affinity == 'LLE' || affinity == 'geo-t-SNE'
                                        时，作为算法所需的参数
                        'neighborhood_size': 局部邻域的半径大小，当 neighborhood_type == 'rnn' 时有效
                        'distance_type': 协方差矩阵的距离标准
                                        'spectralNorm': 谱范数，直接使用 numpy.linalg.norm(Matrix)
                                        'mahalanobis': 马氏距离，对差矩阵进行特征值分解，取最大的特征值（假设所有特征值均为非负数）
                        'manifold_dimension': 流形本身的维度
                        'perplexity': 用 t-SNE 降维时的困惑度，当 affinity == 't-SNE' || affinity == 'geo-t-SNE' 时有效
                                    当 frame == 't-SNE' 时，用于控制方差大小
                        'MAX_Distance_iter': 迭代式计算距离矩阵时，最多的迭代次数
                        'use_skeleton': 是否使用骨架点的方法，取值为 True 或 False
                        'save_path': 用于存储中间临时结果的路径
        :param frame: 迭代求解降维结果时用的框架
                            'MDS': 使用 SAMCOF 算法求解
                            't-SNE': 使用 t-SNE 的迭代方式求解
                            't-SNE+': 使用 sklearn 中的加速版 t-SNE 求解
        """
        self.n_components = n_components
        self.affinity = affinity
        self.parameters = parameters
        self.frame = frame
        self.skeleton_Y = None  # 骨架点的降维结果
        self.path = parameters['save_path']  # 存储中间结果的路径
        self.config_str = '[n_neighbors=' + str(parameters['n_neighbors']) + " alpha=" + str(parameters['alpha']) + ']'

    def local_cov(self, X):
        """
        计算每个点的 local covariance matrix
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return: 一个三维的张量，其中 M[i, :, :] 为第 i 个点的 local covariance matrix
        """
        (n, m) = X.shape
        M = np.zeros((n, m, m))

        if self.parameters['neighborhood_type'] == 'knn' or self.parameters['neighborhood_type'] == 'iter':
            print("Calculate local covariance matrix using KNN...")
            M = local_cov_knn(X, self.parameters['n_neighbors'])
        elif self.parameters['neighborhood_type'] == 'rnn':
            print("Calculate local covariance matrix using RNN...")
            M = local_cov_rnn(X, self.parameters['neighborhood_size'])
        # elif self.parameters['neighborhood_type'] == 'distanceKNN':
        #     print('迭代计算')
        else:
            print("Wrong neighborhood_type: " + str(self.parameters['neighborhood_type']))
            return

        return M

    def Q_matrix(self, Cov):
        """
        用奇异值分解的方法重建 Q 矩阵
        :param Cov: 每个点的协方差矩阵
        :return:
        """
        (n, m, k) = Cov.shape
        Q = np.zeros(Cov.shape)
        d = self.parameters['manifold_dimension']
        sum_S = np.zeros((m, ))

        for i in range(0, n):
            (U, S, V) = np.linalg.svd(Cov[i, :, :])
            if d > 1:
                Q[i, :, :] = np.matmul(U[:, 0:d], U[:, 0:d].T)
            else:
                Q[i, :, :] = np.outer(U[:, 0], U[:, 0])

            sum_S = sum_S + S / np.sum(S)

        sum_S = sum_S / n
        # 画每个奇异值的占比
        plt.plot(sum_S)
        plt.title("eigenvalues")
        plt.show()
        print("每个特征值的占比：")
        print(sum_S)

        # 前 i 个奇异值的占比
        top_sum = np.zeros((m, ))
        top_sum[:] = sum_S[:]
        for i in range(0, m):
            top_sum[i] = top_sum[i] + top_sum[i-1]

        plt.plot(top_sum)
        plt.title("top eigenvalues")
        plt.show()

        return Q

    def iter_affinity_matrix(self, X):
        """
        迭代式的方式计算距离关系
        :param X: 高维数据矩阵
        :return:
        """
        W = self.affinity_matrix(X)
        euclidean = euclidean_distances(X)
        count = 1
        knn = distanceIter.neighbors_by_distance(W, self.parameters['n_neighbors'])
        while count < self.parameters['MAX_Distance_iter']:
            Cov = local_cov_by_knn(X, knn)
            W = self.affinity_matrix_sub(euclidean, Cov)
            knn2 = distanceIter.neighbors_by_distance(W, self.parameters['n_neighbors'])
            count += 1
            if distanceIter.knn_equals(knn, knn2):
                break
            else:
                knn = knn2.copy()
        return W

    def affinity_matrix_sub(self, euclidean, Cov):
        """
        根据欧氏距离矩阵与local PCA相似度矩阵计算每个点之间的相似性
        :param euclidean: 欧氏距离矩阵
        :param Cov: 里面存储的是每个点的 local covariance matrix
        :return:
        """
        (n, n1) = euclidean.shape
        if self.affinity == 'cov':
            print('Calculate the spectral distance of local covariance matrix...')
            Cd = cov_matrix_distance(Cov, self.parameters['distance_type'])  # 协方差矩阵之间的谱距离
            Cd = Cd / (np.max(Cd) + 1e-15)
            W = self.parameters["alpha"] * euclidean + self.parameters["beta"] * Cd
            return W
        elif self.affinity == 'Q':
            Q = self.Q_matrix(Cov)
            print('Calculate the spectral distance of local projection matrix...')
            Qd = cov_matrix_distance(Q, self.parameters['distance_type'])
            Qd = Qd / (np.max(Qd) + 1e-15)
            W = self.parameters['alpha'] * euclidean + self.parameters['beta'] * Qd
            return W
        else:
            print('暂不支持该方法')
            return

    def affinity_matrix(self, X, Cov=None):
        """
        计算 affinity 矩阵
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :param Cov: 张量，存储每个点的协方差矩阵。如果不是 None，需要重新计算
        :return:
        """
        (n, m) = X.shape
        W = np.zeros((n, n))
        print('Calculate Euclidean distance...')
        Ed = euclidean_distances(X)  # 欧氏距离矩阵
        Ed = Ed / (np.max(Ed) + 1e-15)

        if self.affinity == 'cov':
            # 协方差矩阵距离与欧氏距离加权和
            if Cov is None:
                Cov = self.local_cov(X)  # 协方差矩阵
            print('Calculate the spectral distance of local covariance matrix...')
            Cd = cov_matrix_distance(Cov, self.parameters['distance_type'])  # 协方差矩阵之间的谱距离
            Cd = Cd / (np.max(Cd) + 1e-15)
            W = self.parameters["alpha"] * Ed + self.parameters["beta"] * Cd
            return W
        elif self.affinity == 'expCov':
            # 综合考虑协方差矩阵的距离与欧氏距离，然后用 exp 函数加工
            # 协方差矩阵距离与欧氏距离加权和
            if Cov is None:
                Cov = self.local_cov(X)  # 协方差矩阵
            print('Calculate the spectral distance of local covariance matrix...')
            Cd = cov_matrix_distance(Cov, self.parameters['distance_type'])  # 协方差矩阵之间的谱距离
            Cd = Cd / (np.max(Cd) + 1e-15)
            W = self.parameters["alpha"] * Ed + self.parameters["beta"] * Cd
            W = np.exp(W)  # 这里每个点的方差设置成了完全相同的，可能还是会需要设置不同的方差
            return W
        elif self.affinity == 'Q':
            # Q 矩阵的相似性 加上 欧氏距离的相似性
            if Cov is None:
                Cov = self.local_cov(X)  # 协方差矩阵
            print('Calculate Q matrix...')
            Q = self.Q_matrix(Cov)
            print('Calculate the spectral distance of local projection matrix...')
            Qd = cov_matrix_distance(Q, self.parameters['distance_type'])
            Qd = Qd / (np.max(Qd) + 1e-15)
            W = self.parameters['alpha'] * Ed + self.parameters['beta'] * Qd
            np.savetxt(self.path + self.config_str + "final_distance_matrix" + ".csv", W, fmt='%.18e', delimiter=",")
            np.savetxt(self.path + self.config_str + "euclidean_distance_matrix" + ".csv", W, fmt='%.18e', delimiter=",")
            return W
        elif self.affinity == 'expQ':
            # 综合考虑投影矩阵 Q 与欧氏距离，然后用 exp 函数进行加工
            if Cov is None:
                Cov = self.local_cov(X)  # 协方差矩阵
            print('Calculate Q matrix...')
            Q = self.Q_matrix(Cov)
            print('Calculate the spectral distance of local projection matrix...')
            Qd = cov_matrix_distance(Q, self.parameters['distance_type'])
            Qd = Qd / (np.max(Qd) + 1e-15)
            W = self.parameters['alpha'] * Ed + self.parameters['beta'] * Qd
            W = np.exp(W)
            return W

        return W

    def skeleton_fit_transform(self, X):
        """
        使用骨架点的计算
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return:
        """
        (n, m) = X.shape
        print(self.parameters)

        skeleton, satellite = skeletonMethod.get_skeleton(X,
                                                          neighborhood_type=self.parameters['neighborhood_type'],
                                                          n_neighbors=self.parameters['n_neighbors'],
                                                          neighborhood_size=self.parameters['neighborhood_size'],
                                                          label=None)
        print('Use skeleton method, there are ' + str(n) + ' points in total, and ' + str(len(skeleton))
              + ' are skeleton points.')
        Cov = skeletonMethod.get_skeleton_cov(X, satellite)  # 每个骨架点的协方差矩阵
        skeleton_X = np.zeros((len(skeleton), m))
        index = 0
        for s in skeleton:
            skeleton_X[index, :] = X[s, :]
            index = index + 1
        W = self.affinity_matrix(skeleton_X, Cov)  # 骨架点之间的距离度量

        skeleton_Y = np.zeros((len(skeleton), self.n_components))  # 骨架点的降维结果
        if self.frame == 'MDS':
            print('Using MDS frame...')
            mds = MDS(n_components=self.n_components, dissimilarity='precomputed')
            skeleton_Y = mds.fit_transform(W)
        elif self.frame == 't-SNE':
            print('Using t-SNE frame...')
            skeleton_Y = tsneFrame.tsne_plus(W, self.parameters['perplexity'], path=self.path, config_str=self.config_str)
        elif self.frame == 't-SNE+':
            print('Using t-SNE framework in sklearn...')
            tsne = tsneFramePlus.tsnePlus(n_components=self.n_components, perplexity=self.parameters['perplexity'])
            Y = tsne.fit_transform(W)
            return Y
        else:
            print("Wrong frame name!")
            return
        self.skeleton_Y = skeleton_Y  # 这里保存一下，用于画骨架点的坐标看看

        Y = skeletonMethod.satellite_location(X, skeleton, skeleton_Y)  # 所有点的降维结果
        return Y

    def fit_transform(self, X):
        """
        计算降维结果
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return:
        """
        (n, m) = X.shape
        print(self.parameters)

        # 用经典的降维方法
        if self.affinity == 'PCA':  # 直接返回 PCA 的降维结果
            print('Classical method: PCA...')
            pca = PCA(n_components=self.n_components)
            return pca.fit_transform(X)
        elif self.affinity == 'MDS':  # 直接返回 MDS 的降维结果
            print('Classical method: MDS...')
            mds = MDS(n_components=self.n_components)
            return mds.fit_transform(X)
        elif self.affinity == 'Isomap':  # 直接返回 Isomap 的降维结果
            print('Classical method: Isomap...')
            iso = Isomap(n_components=self.n_components, n_neighbors=self.parameters['n_neighbors'])
            return iso.fit_transform(X)
        elif self.affinity == 't-SNE':  # 直接返回 t-SNE 的降维结果
            print('Classical method: t-SNE...')
            tsne = TSNE(n_components=self.n_components, perplexity=self.parameters['perplexity'])
            return tsne.fit_transform(X)
        elif self.affinity == 'cTSNE':  # 用不加速版本的t-SNE降维
            print('Classical method: classical t-SNE...')
            from ArtDR import tsne
            return tsne.tsne(X, perplexity=self.parameters['perplexity'], path=self.path, config_str='t-SNE ')
        elif self.affinity == 'LLE':  # 直接返回 LLE 的降维结果
            print('Classical method: LLE...')
            lle = LocallyLinearEmbedding(n_components=self.n_components, n_neighbors=self.parameters['n_neighbors'])
            return lle.fit_transform(X)
        elif self.affinity == 'geo-t-SNE':  # 用基于测地线距离的 t-SNE 方法
            print('Geodesic t-SNE...')
            gtsne = geoTsne(n_neighbors=self.parameters['n_neighbors'], perplexity=self.parameters['perplexity'])
            return gtsne.fit_transform(X, n_components=self.n_components)

        if self.parameters['use_skeleton']:  # 用骨架点的方法
            return self.skeleton_fit_transform(X)

        # 用我们自己设计的降维方法
        if self.parameters['neighborhood_type'] == 'iter':  # 用迭代的方式
            W = self.iter_affinity_matrix(X)
        else:
            W = self.affinity_matrix(X)  # 用我们的普通方法
        if self.frame == 'MDS':
            print('Using MDS frame...')
            mds = MDS(n_components=self.n_components, dissimilarity='precomputed')
            Y = mds.fit_transform(W)
            return Y
        elif self.frame == 't-SNE':
            print('Using t-SNE frame...')
            Y = tsneFrame.tsne_plus(W, self.parameters['perplexity'], path=self.path, config_str=self.config_str)
            return Y
        elif self.frame == 't-SNE+':
            print('Using t-SNE framework in sklearn...')
            tsne = tsneFramePlus.tsnePlus(n_components=self.n_components, perplexity=self.parameters['perplexity'])
            Y = tsne.fit_transform(W)
            return Y
        else:
            print("Wrong frame name!")
            return
