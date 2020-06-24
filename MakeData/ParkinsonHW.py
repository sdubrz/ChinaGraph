import numpy as np


def test():
    path = "E:\\ChinaGraph\\DataLab\\PARKINSON_HW\\"
    data = np.loadtxt(path+"test.txt", dtype=np.int, delimiter=";")
    (n, m) = data.shape
    print(data.shape)
    X = data[:, 0:5]
    label = data[:, 6]
    np.savetxt(path+"data.csv", X, fmt='%d', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


if __name__ == '__main__':
    test()
