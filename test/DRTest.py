import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE


def MDS_test(X, label):
    mds = MDS(n_components=2)
    Y = mds.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("MDS")
    plt.show()


def Isomap_test(X, label):
    iso = Isomap(n_components=2, n_neighbors=100)
    Y = iso.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("Isomap")
    plt.show()


def tsne_test(X, label):
    tsne = TSNE(n_components=2, perplexity=10.0)
    Y = tsne.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("t-SNE")
    plt.show()


if __name__ == '__main__':
    path = "E:\\ChinaGraph\\Test\\dataset\\2plane90\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    # MDS_test(X, label)
    # Isomap_test(X, label)
    tsne_test(X, label)

