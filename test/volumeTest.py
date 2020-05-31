import numpy as np


def test1():
    path = "E:\\ChinaGraph\\Test\\volume\\CLOUDf01.bin"
    X = np.loadtxt(path)
    print(X.shape)


if __name__ == '__main__':
    test1()
