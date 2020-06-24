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


def combination01():
    path = "E:\\ChinaGraph\\DataLab\\MNIST68\\"
    data0 = np.loadtxt(path+"6.csv", dtype=np.int, delimiter=",")
    data1 = np.loadtxt(path+"8.csv", dtype=np.int, delimiter=",")
    (n0, m) = data0.shape
    (n1, m) = data1.shape
    label0 = np.zeros((n0, 1))
    label1 = np.ones((n1, 1))

    X0 = data0 - np.mean(data0, axis=0)
    X1 = data1 - np.mean(data1, axis=0)
    np.savetxt(path+"data0.csv", X0, fmt='%d', delimiter=",")
    np.savetxt(path+"data1.csv", X1, fmt='%d', delimiter=",")
    np.savetxt(path+"label0.csv", label0, fmt='%d', delimiter=",")
    np.savetxt(path+"label1.csv", label1, fmt='%d', delimiter=",")


def big_combination():
    # 将所有的 10 个类分别中心化，并全部组合起来
    data_list = []
    label_list = []
    count = np.zeros((10, 1))
    path = "E:\\ChinaGraph\\DataLab\\MNISTtest\\classify\\"
    X = np.zeros((10000, 784))
    label = np.zeros((10000, 1))
    for i in range(0, 10):
        data_i = np.loadtxt(path+"data"+str(i)+".csv", dtype=np.int, delimiter=",")
        label_i = np.loadtxt(path+"label"+str(i)+".csv", dtype=np.int, delimiter=",")
        (n_i, m_i) = data_i.shape
        count[i] = n_i
        if i == 0:
            X[0:n_i, :] = data_i - np.mean(data_i, axis=0)
            label[0:n_i] = 0
        else:
            head = int(np.sum(count[0:i]))
            print("head = ", head)
            X[head:head+n_i, :] = data_i - np.mean(data_i, axis=0)
            label[head:head+n_i] = i
        print(i)

    np.savetxt(path+"data.csv", X, fmt='%d', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


def classify():
    """
    按照类别划分
    :return:
    """
    path = "E:\\ChinaGraph\\DataLab\\MNISTtest\\"
    data0 = np.loadtxt(path + "data.csv", dtype=np.int, delimiter=",")
    label0 = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, m) = data0.shape
    print(data0.shape)

    n_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, n):
        n_list[label0[i]] = n_list[label0[i]] + 1

    for c in range(0, 10):
        n1 = n_list[c]
        label = c * np.ones((n1, 1))
        data = np.zeros((n1, m))
        count = 0
        for i in range(0, n):
            if label0[i] == c:
                data[count, :] = data0[i, :]
                count += 1
        np.savetxt(path+"classify\\label"+str(c)+".csv", label, fmt='%d', delimiter=",")
        np.savetxt(path+"classify\\data"+str(c)+".csv", data, fmt='%d', delimiter=",")
        print(c)


if __name__ == '__main__':
    # test()
    # run()
    # combination01()
    # classify()
    big_combination()


