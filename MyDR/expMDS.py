import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


class expMDS:
    def __init__(self, n_component=2):
        self.n_component = n_component
        self.W = None  # 不相似性矩阵

    def fit_transform(self, X):
        """
        执行降维
        :param X: 高维数据矩阵，每一行是一个高维数据点
        :return:
        """
        (n, m) = X.shape
        W = np.zeros((n, n))
        for i in range(0, n):
            for j in range(i+1, n):
                dx = X[i, :] - X[j, :]
                dot = np.dot(dx, dx)
                exp = 1-np.exp(-dot)
                W[i, j] = exp
                W[j, i] = exp
        self.W = W
        mds = MDS(n_components=2, dissimilarity='precomputed')
        Y = mds.fit_transform(W)
        return Y


if __name__ == '__main__':
    path = "E:\\ChinaGraph\\Data\\IsomapFace\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    obj = expMDS(n_component=2)
    Y = obj.fit_transform(data)
    np.savetxt(path+"W.csv", obj.W, fmt='%.18e', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

