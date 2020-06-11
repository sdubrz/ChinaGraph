import numpy as np
from sklearn.manifold import TSNE


# 加速版的 t-SNE 框架，调用 sklearn 中的实现


class tsnePlus:
    def __init__(self, n_components=2, perplexity=30.0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.tsne = TSNE(n_components=n_components, metric='precomputed', perplexity=perplexity)

    def fit_transform(self, W):
        """
        执行降维
        :param W: 距离矩阵，是一个方阵
        :return:
        """
        Y = self.tsne.fit_transform(W)
        return Y

