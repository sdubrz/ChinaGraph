import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MakeData import DartsSampling
import random


path = 'E:\\ChinaGraph\\Data\\temp\\'


def swissroll():
    """
    生成 swissroll 数据
    :return:
    """
    max_fail = 3000  # 最大失败次数

    data = []
    points = []
    loop_count = 0
    while loop_count < max_fail:
        t = random.uniform(0, 1)
        t = t * np.pi * 3 + np.pi
        virtual_x = (t * t - np.pi * np.pi) / 2
        temp_z = random.uniform(0, 1) * 15
        p = [virtual_x, temp_z]
        if DartsSampling.all_far(points, p, radius=0.85):
            points.append(p)
            temp_x = t * np.sin(t)
            temp_y = t * np.cos(t)
            data.append([temp_x, temp_y, temp_z])
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count += 1

    n = len(points)
    print(n)
    X = np.array(data)

    np.savetxt(path + "data.csv", X, fmt='%f', delimiter=",")
    np.savetxt(path + "label.csv", np.ones((n, 1)), fmt='%d', delimiter=",")
    ax = Axes3D(plt.figure())
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.show()


if __name__ == '__main__':
    swissroll()
