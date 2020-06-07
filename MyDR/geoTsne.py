import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
# 使用测地线距离的 t-SNE 方法


def geodesic_distance(X, n_neighbors):
    """
    计算 X 中各个点之间的测地线距离
    :param X: 数据矩阵，每一行是一个点
    :param n_neighbors: 邻居数，这些点可以直接用欧氏距离来当做测地线距离
    :return:
    """
    (n, m) = X.shape
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    nbrs_dist, knn = nbrs.kneighbors(X)

    distance = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n_neighbors):
            distance[i, knn[i, j]] = nbrs_dist[i, j]
            distance[knn[i, j], i] = nbrs_dist[i, j]
    D = graph_shortest_path(distance, directed=False)
    return D


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(distance=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
        :param distance: 点与点之间的欧氏距离矩阵
        :param tol:
        :param perplexity:
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, n1) = distance.shape
    D = distance ** 2
    # (n, d) = X.shape
    # sum_X = np.sum(np.square(X), 1)
    # D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


class geoTsne:
    def __init__(self, n_neighbors=5, perplexity=30.0):
        """
        初始化方法
        :param n_neighbors: 邻居点数
        :param perplexity: 困惑度
        """
        self.n_neighbors = n_neighbors
        self.perplexity = perplexity

    def fit_transform(self, X, n_components=2):
        """
        执行降维过程
        :param X: 高维数据矩阵，每一行是一个数据点
        :param n_components: 降维之后的维度
        :return:
        """
        (n, m) = X.shape
        D = geodesic_distance(X, self.n_neighbors)

        max_iter = 1000
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01
        Y = np.random.randn(n, n_components)
        dY = np.zeros((n, n_components))
        iY = np.zeros((n, n_components))
        gains = np.ones((n, n_components))

        # Compute P-values
        P = x2p(D, 1e-5, self.perplexity)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4.  # early exaggeration
        P = np.maximum(P, 1e-12)

        # Run iterations
        for iter in range(max_iter):

            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Compute gradient
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))

            # Stop lying about P-values
            if iter == 100:
                P = P / 4.

        # Return solution
        return Y


def run_example():
    path = 'E:\\ChinaGraph\\Data\\2plane90\\'
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    obj = geoTsne(n_neighbors=8, perplexity=30.0)
    Y = obj.fit_transform(X, n_components=2)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()


if __name__ == '__main__':
    run_example()

