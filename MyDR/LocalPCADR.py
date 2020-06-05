import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from MyDR import tsneFrame


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
        for j in range(i+1, n):
            if distance_type == 'spectralNorm':
                d = np.linalg.norm(Cov[i, :, :]-Cov[j, :, :])
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
    def __init__(self, n_components=2, affinity='expCov', parameters={}, frame='MDS', manifold_dimension=2):
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
                        'distance_type': 协方差矩阵的距离标准
                                        'spectralNorm': 谱范数，直接使用 numpy.linalg.norm(Matrix)
                                        'mahalanobis': 马氏距离，对差矩阵进行特征值分解，取最大的特征值（假设所有特征值均为非负数）
                        'manifold_dimension': 流形本身的维度
        :param frame: 迭代求解降维结果时用的框架
                            'MDS': 使用 SAMCOF 算法求解
                            't-SNE': 使用 t-SNE 的迭代方式求解
        :param manifold_dimension: 流形本身的维度数
        """
        self.n_components = n_components
        self.affinity = affinity
        self.parameters = parameters
        self.frame = frame

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
            M = local_cov_knn(X, self.parameters['n_neighbors'])
        elif self.parameters['neighborhood_type'] == 'rnn':
            print("Calculate local covariance matrix using RNN...")
            M = local_cov_rnn(X, self.parameters['neighborhood_size'])
        else:
            print("Wrong neighborhood_type: "+str(self.parameters['neighborhood_type']))
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

        for i in range(0, n):
            (U, S, V) = np.linalg.svd(Cov[i, :, :])
            if d > 1:
                Q[i, :, :] = np.matmul(U[:, 0:d], U[:, 0:d].T)
            else:
                Q[i, :, :] = np.outer(U[:, 0], U[:, 0])

        return Q

    def affinity_matrix(self, X):
        """
        计算 affinity 矩阵
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return:
        """
        (n, m) = X.shape
        W = np.zeros((n, n))
        print('Calculate Euclidean distance...')
        Ed = euclidean_distances(X)  # 欧氏距离矩阵
        Ed = Ed / (np.max(Ed)+1e-15)

        if self.affinity == 'cov':
            # 协方差矩阵距离与欧氏距离加权和
            Cov = self.local_cov(X)  # 协方差矩阵
            print('Calculate the spectral distance of local covariance matrix...')
            Cd = cov_matrix_distance(Cov, self.parameters['distance_type'])  # 协方差矩阵之间的谱距离
            Cd = Cd / (np.max(Cd)+1e-15)
            W = self.parameters["alpha"] * Ed + self.parameters["beta"] * Cd
            return W
        elif self.affinity == 'expCov':
            # 综合考虑协方差矩阵的距离与欧氏距离，然后用 exp 函数加工
            # 协方差矩阵距离与欧氏距离加权和
            Cov = self.local_cov(X)  # 协方差矩阵
            print('Calculate the spectral distance of local covariance matrix...')
            Cd = cov_matrix_distance(Cov, self.parameters['distance_type'])  # 协方差矩阵之间的谱距离
            Cd = Cd / (np.max(Cd) + 1e-15)
            W = self.parameters["alpha"] * Ed + self.parameters["beta"] * Cd
            W = np.exp(W)  # 这里每个点的方差设置成了完全相同的，可能还是会需要设置不同的方差
            return W
        elif self.affinity == 'Q':
            # Q 矩阵的相似性 加上 欧氏距离的相似性
            Cov = self.local_cov(X)  # 协方差矩阵
            print('Calculate Q matrix...')
            Q = self.Q_matrix(Cov)
            print('Calculate the spectral distance of local projection matrix...')
            Qd = cov_matrix_distance(Q, self.parameters['distance_type'])
            Qd = Qd / (np.max(Qd)+1e-15)
            W = self.parameters['alpha'] * Ed + self.parameters['beta'] * Qd
            return W
        elif self.affinity == 'expQ':
            # 综合考虑投影矩阵 Q 与欧氏距离，然后用 exp 函数进行加工
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

    def fit_transform(self, X):
        """
        计算降维结果
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return:
        """
        (n, m) = X.shape
        print(self.parameters)

        W = self.affinity_matrix(X)
        if self.frame == 'MDS':
            print('Using MDS frame...')
            mds = MDS(n_components=self.n_components, dissimilarity='precomputed')
            Y = mds.fit_transform(W)
            return Y
        elif self.frame == 't-SNE':
            print('Using t-SNE frame...')
            Y = tsneFrame.tsne_plus(W)
            return Y
        else:
            print("Wrong frame name!")
            return


def run_example():
    """
    一个使用 local PCA 降维方法的示例
    :return:
    """
    path = "E:\\ChinaGraph\\Data\\2plane90\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    # 如果是三维的，则画出三维散点图
    ax3d = Axes3D(plt.figure())
    ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=label)
    plt.title('original data')
    plt.show()

    params = {}
    params['neighborhood_type'] = 'knn'  # 'knn' or 'rnn'
    params['n_neighbors'] = 10  # Only used when neighborhood_type is 'knn'
    params['neighborhood_size'] = 1.0  # Only used when neighborhood_type is 'rnn'
    params['alpha'] = 0.5  # the weight of euclidean distance
    params['beta'] = 1 - params['alpha']  # the weight of local PCA
    params['distance_type'] = 'spectralNorm'  # 'spectralNorm' or 'mahalanobis'
    params['manifold_dimension'] = 2  # the real dimension of manifolds
    affinity = 'cov'  # affinity 的取值可以为 'cov'  'expCov'  'Q'  'expQ'
    frame_work = 'MDS'  # frame 的取值可以为 'MDS'  't-SNE'
    dr = LocalPCADR(n_components=2, affinity=affinity, parameters=params, frame=frame_work, manifold_dimension=2)

    Y = dr.fit_transform(X)

    # 画图
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    title_str = 'Frame[' + frame_work + '] ' + affinity + ' alpha=' + str(params['alpha']) + ' beta=' + str(params['beta'])
    if params['neighborhood_type'] == 'knn':
        title_str = title_str + ' k=' + str(params['n_neighbors'])
    elif params['neighborhood_type'] == 'rnn':
        title_str = title_str + ' r=' + str(params['neighborhood_size'])
    plt.title(title_str)
    plt.show()


if __name__ == '__main__':
    run_example()
