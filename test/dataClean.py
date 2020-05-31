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


if __name__ == '__main__':
    plane_make()
