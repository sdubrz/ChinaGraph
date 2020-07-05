# 处理fashion数据
import numpy as np


def test1():
    path = "E:\\ChinaGraph\\DataLab\\fashion\\"
    total = np.loadtxt(path+"fashion.csv", dtype=np.int, delimiter=",")
    (n, m) = total.shape
    print(total.shape)

    label = np.zeros((5000, 1))
    data = np.zeros((5000, 784))
    count = 0
    for i in range(0, n):
        if i%2 == 0:
            label[count] = total[i, 0]
            data[count, :] = total[i, 1:m]
            count = count + 1

    np.savetxt(path+"origin.csv", data, fmt='%d', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


def classify():
    """
    按照类别划分
    :return:
    """
    path = "E:\\ChinaGraph\\DataLab\\fashion\\"
    total = np.loadtxt(path + "fashion.csv", dtype=np.int, delimiter=",")
    (n, m) = total.shape
    print(total.shape)

    n_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, n):
        n_list[total[i, 0]] = n_list[total[i, 0]] + 1

    for c in range(0, 10):
        n1 = n_list[c]
        label = c * np.ones((n1, 1))
        data = np.zeros((n1, m-1))
        count = 0
        for i in range(0, n):
            if total[i, 0] == c:
                data[count, :] = total[i, 1:m]
                count += 1
        np.savetxt(path+"classify\\label"+str(c)+".csv", label, fmt='%d', delimiter=",")
        np.savetxt(path+"classify\\data"+str(c)+".csv", data, fmt='%d', delimiter=",")
        print(c)


def center():
    path1 = "E:\\ChinaGraph\\DataLab\\fashionCenter\\classify\\"
    path2 = "E:\\ChinaGraph\\DataLab\\fashionCenter\\center\\"
    for i in range(0, 10):
        X = np.loadtxt(path1+"data"+str(i)+".csv", dtype=np.int, delimiter=",")
        X = X - np.mean(X, axis=0)
        np.savetxt(path2+"data"+str(i)+".csv", X, fmt='%d', delimiter=",")
        print(i)


def combine():
    path = "E:\\ChinaGraph\\DataLab\\fashionCenter\\"
    data = np.zeros((10000, 784))
    origin = np.zeros((10000, 784))
    label = np.zeros((10000, 1))
    ptr = 0

    for i in range(0, 10):
        X = np.loadtxt(path+"center\\data"+str(i)+".csv", dtype=np.int, delimiter=",")
        sub_origin = np.loadtxt(path+"classify\\data"+str(i)+".csv", dtype=np.int, delimiter=",")
        (n, m) = X.shape
        data[ptr:ptr+n, :] = X[:, :]
        origin[ptr:ptr+n, :] = sub_origin[:, :]
        label[ptr:ptr+n] = i
        ptr = ptr + n
    np.savetxt(path+"data.csv", data, fmt='%d', delimiter=",")
    np.savetxt(path+"origin.csv", origin, fmt='%d', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


if __name__ == '__main__':
    # test1()
    # classify()
    center()
    combine()

