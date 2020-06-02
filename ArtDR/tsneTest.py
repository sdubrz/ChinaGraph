import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# 寻找 t-SNE 降维效果不好的例子


def run():
    path = "E:\\ChinaGraph\\Data\\hybrid\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    X = data

    tsne = TSNE(n_components=2, perplexity=30.0)
    Y = tsne.fit_transform(X)

    ax1 = plt.subplot


if __name__ == '__main__':
    run()


