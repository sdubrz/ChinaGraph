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


if __name__ == '__main__':
    test1()

