import numpy as np
# 处理 YaleFace 数据集


def center():
    """
    做中心化处理， 其他数据也可以直接使用这个方法
    :return:
    """
    path = "E:\\ChinaGraph\\DataLab\\YaleFaceCenter\\"
    data0 = np.loadtxt(path+"data0.csv", dtype=np.int, delimiter=",")
    label0 = np.loadtxt(path+"label0.csv", dtype=np.int, delimiter=",")
    origin0 = np.loadtxt(path+"origin0.csv", dtype=np.int, delimiter=",")
    (n, m) = data0.shape

    data = np.zeros(data0.shape)
    label = np.zeros((n, 1))
    origin = np.zeros(origin0.shape)

    label_list = []
    for i in range(0, n):
        if not (label0[i] in label_list):
            label_list.append(label0[i]) 

    ptr = 0
    for cluster in label_list:
        print(cluster)
        X_list = []
        origin_list = []
        for i in range(0, n):
            if label0[i] == cluster:
                X_list.append(data0[i, :])
                origin_list.append(origin0[i, :])
        ni = len(X_list)
        X = np.array(X_list)
        origin_i = np.array(origin_list)

        X = X - np.mean(X, axis=0)
        data[ptr:ptr+ni, :] = X[:, :]
        label[ptr:ptr+ni] = cluster
        origin[ptr:ptr+ni, :] = origin_i[:, :]
        ptr = ptr + ni

    np.savetxt(path+"data.csv", data, fmt='%d', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")
    np.savetxt(path+"origin.csv", origin, fmt='%d', delimiter=",")


if __name__ == '__main__':
    center()

