import numpy as np


def make_label():
    path = "E:\\ChinaGraph\\Data\\OlivettiFaces\\"
    label = np.zeros((400, 1))

    for i in range(0, 40):
        for j in range(0, 10):
            label[i*10+j] = i
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


if __name__ == '__main__':
    make_label()

