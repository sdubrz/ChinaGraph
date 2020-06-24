import numpy as np
import random


# 给MNIST数据添加随机噪声查看效果
def run():
    path = "E:\\ChinaGraph\\DataLab\\MNISTnoise\\"
    data0 = np.loadtxt(path+"0.csv", dtype=np.int, delimiter=",")
    data1 = np.loadtxt(path+"1.csv", dtype=np.int, delimiter=",")
    alpha = 0.7  # 控制随机程度大小

    (n0, m) = data0.shape
    (n1, m) = data1.shape
    label = np.zeros((n1+n0, 1))
    label[n0:n0+n1] = 1
    X = np.zeros((n0+n1, m))
    X[0:n0, :] = data0[:, :] + np.random.rand(n0, m) * 255 * alpha
    X[n0:n0+n1, :] = data1[:, :] + np.random.rand(n1, m) * 255 * alpha
    np.savetxt(path+"data.csv", X, fmt='%d', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


def test():
    X = np.random.rand(3, 4)
    print(X)


if __name__ == '__main__':
    # test()
    run()


