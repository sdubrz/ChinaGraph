import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from MyDR import Similarity


# 修改MDS降维方法
def local_mds(X, n_neighbors=5, alpha=0.32):
    """
    增加了local因素的MDS实现
    :param X:
    :param n_neighbors:
    :param alpha:
    :return:
    """
    (n, m) = X.shape
    mds = MDS(n_components=2, dissimilarity='precomputed')
    D = Similarity.distance_and_cov(X, n_neighbors=n_neighbors, alpha=alpha)  # 相加的版本
    # D = Similarity.distance_multi_cov(X, n_neighbors=n_neighbors)  # 相乘的版本

    print("gradient descent...")
    Y = mds.fit_transform(D)
    return Y


def run_test():
    path = "E:\\ChinaGraph\\Data\\"
    data_name = "2plane90"
    data_path = path + data_name+"\\"
    data = np.loadtxt(data_path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(data_path+"label.csv", dtype=np.int, delimiter=",")

    n_negighbors = 10
    alpha = 0.8

    # 用现有的降维方法做对比
    # from MyDR import ArtMethod
    # Y_pca = ArtMethod.run_pca(data, label)
    # np.savetxt(data_path + "PCA.csv", Y_pca, fmt='%.18e', delimiter=",")
    # Y_mds = ArtMethod.run_mds(data, label)
    # np.savetxt(data_path+"MDS.csv", Y_mds, fmt='%.18e', delimiter=",")
    # Y_iso = ArtMethod.run_isomap(data, label, n_neighbors=n_negighbors)
    # np.savetxt(data_path + "Isomap.csv", Y_iso, fmt='%.18e', delimiter=",")
    # Y_lle = ArtMethod.run_lle(data, label, n_neighbors=n_negighbors)
    # np.savetxt(data_path + "LLE.csv", Y_lle, fmt='%.18e', delimiter=",")
    # Y_tsne = ArtMethod.run_tsne(data, label, perplexity=30.0)
    # np.savetxt(data_path + "tSNE.csv", Y_tsne, fmt='%.18e', delimiter=",")

    Y = local_mds(data, n_neighbors=n_negighbors, alpha=alpha)
    np.savetxt(data_path+"AddCov_neighbors="+str(n_negighbors)+"_alpha="+str(alpha)+".csv", Y, fmt='%.18e', delimiter=",")
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("cov, neighbors="+str(n_negighbors)+", alpha="+str(alpha))
    plt.show()


if __name__ == '__main__':
    run_test()

