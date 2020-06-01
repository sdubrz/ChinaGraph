import numpy as np
import matplotlib.pyplot as plt


def Wtest():
    path = "E:\\ChinaGraph\\SMMCW\\hybrid\\"
    W = np.loadtxt(path + "W.csv", dtype=np.float, delimiter=",")
    W = np.max(W) - W
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")

    (n, m) = W.shape
    W = np.maximum(W, 1e-12)
    for i in range(0, n):
        temp_sum = np.sum(W[i, :])
        W[i, :] = W[i, :] / temp_sum

    from MyDR import NewTsne
    W = np.maximum(W, 1e-12)
    Y = NewTsne.tsne_plus(W)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()


if __name__ == '__main__':
    Wtest()
