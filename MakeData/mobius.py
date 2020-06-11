import numpy as np


def sampling():
    path = "E:\\ChinaGraph\\Data\\mobius\\"
    X0 = np.loadtxt(path+"data0.csv", dtype=np.float, delimiter=",")
    label0 = np.loadtxt(path+"label0.csv", dtype=np.int, delimiter=",")
    (n, d) = X0.shape

    X = []
    label = []
    for i in range(0, n):
        if i % 3 == 0:
            X.append(X0[i, :])
            label.append(label0[i])

    print(len(X))
    np.savetxt(path+"data.csv", X, fmt='%f', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


if __name__ == '__main__':
    sampling()


