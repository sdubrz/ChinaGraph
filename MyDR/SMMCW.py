import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# 直接使用SMMC的相似度矩阵作为MDS的相似度矩阵进行降维


def MDS_SMMC(W):
    """
    用 SMMC 的相似矩阵作为 MDS 的相似度矩阵
    :param W: 用SMMC方法得到的相似度矩阵
    :return:
    """
    mds = MDS(n_components=2, dissimilarity='precomputed')

    print("gradient descent...")
    Y = mds.fit_transform(W)
    return Y


def run_test():
    """
    用数据运行测试
    :return:
    """
    path = "E:\\ChinaGraph\\SMMCW\\hybrid\\"
    W = np.loadtxt(path+"W.csv", dtype=np.float, delimiter=",")
    W = np.max(W) - W
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    Y = MDS_SMMC(W)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    run_test()
