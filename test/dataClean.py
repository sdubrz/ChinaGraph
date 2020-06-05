import numpy as np


def plane_make():
    path = "E:\\ChinaGraph\\Data\\2plane4\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape
    label = np.zeros((n, 1))

    for i in range(0, 549):
        if X[i, 0] > 0.5:
            label[i] = 2
        else:
            label[i] = 1

    for i in range(549, 1014):
        if X[i, 2] > 0.5:
            label[i] = 3
        else:
            label[i] = 4

    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


def norm_test():
    # A = np.array([[1, 2, 3],
    #               [4, 5, 6],
    #               [7, 8, 9]])

    A = np.array([[-1, 1, 0],
                  [-4, 3, 0],
                  [1, 0, 2]])

    print(np.linalg.norm(A, ord=2))

    (values, vectors) = np.linalg.eig(A)
    print(vectors)
    print(values)


def rnn_test():
    """
    测试 RNN 的使用方法
    :return:
    """
    from sklearn.neighbors import NearestNeighbors
    samples = np.array([[0., 0., 0.], [0., .5, 0.], [1., 1., .5]])
    neigh = NearestNeighbors(radius=1.6).fit(samples)
    distance, index = neigh.radius_neighbors(samples)
    print(distance)
    print(index)
    print(distance[0])
    print(distance[0])
    print(type(index))
    print(type(index[0]))

    for temp in index:
        print(temp)

    print(len(index))
    print(len(index[0]))

    dicts = {}
    dicts["hello"] = 1
    dicts["hi"] = 2
    print(str(dicts))


def linear_test():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    (U, S, V) = np.linalg.svd(A)
    print(U)
    print(S)
    print(V)
    B = np.outer(A[:, 0], A[:, 0])
    print(B)


if __name__ == '__main__':
    linear_test()
