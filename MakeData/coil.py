import numpy as np


def center():
    path = "E:\\ChinaGraph\\DataLab\\coil20objCenter\\"

    data = np.loadtxt(path+"data0.csv", dtype=np.int, delimiter=",")
    for i in range(0, 20):
        data[i*72:i*72+72, :] = data[i*72:i*72+72, :] - np.mean(data[i*72:i*72+72, :], axis=0)
        print(i)

    np.savetxt(path+"data.csv", data, fmt='%d', delimiter=",")


if __name__ == '__main__':
    center()
