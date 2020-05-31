import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
# 用现有的降维方法降维


def run_pca(X, label):
    print("PCA...")
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("PCA")
    plt.show()
    return Y


def run_mds(X, label):
    print("MDS...")
    mds = MDS(n_components=2)
    Y = mds.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("MDS")
    plt.show()
    return Y


def run_isomap(X, label, n_neighbors):
    print("Isomap")
    iso = Isomap(n_components=2, n_neighbors=n_neighbors)
    Y = iso.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("Isomap, n_neighbors="+str(n_neighbors))
    plt.show()
    return Y


def run_lle(X, label, n_neighbors):
    print("LLE...")
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors)
    Y = lle.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("LLE, n_neighbors="+str(n_neighbors))
    plt.show()
    return Y


def run_tsne(X, label, perplexity=30.0):
    print("t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title("t-SNE, perplexity="+str(perplexity))
    plt.show()
    return Y


