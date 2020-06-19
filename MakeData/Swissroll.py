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
    color_bar = []  # 用于画渐变颜色的数值
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
            color_bar.append(t)
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count += 1

    n = len(points)
    print(n)
    X = np.array(data)
    bars = np.zeros((n, 1))
    for i in range(0, n):
        bars[i] = color_bar[i]

    np.savetxt(path + "data.csv", X, fmt='%f', delimiter=",")
    np.savetxt(path + "label.csv", np.ones((n, 1)), fmt='%d', delimiter=",")
    np.savetxt(path + "bars.csv", bars, fmt='%f', delimiter=",")
    ax = Axes3D(plt.figure())
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plane():
    """
    生成平面的数据，与 swissroll 产生交叉
    :return:
    """
    max_fail = 3000  # 最大失败次数

    data = []
    points = []
    loop_count = 0
    while loop_count < max_fail:
        x = random.uniform(0, 1) * 28 - 14
        y = random.uniform(0, 1) * 28 - 14
        p = [x, y]
        if DartsSampling.all_far(points, p, radius=0.85):
            points.append(p)
            data.append([x, y, 7.0])
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count += 1

    n = len(points)
    print(n)
    X = np.array(data)

    np.savetxt(path + "data2.csv", X, fmt='%f', delimiter=",")
    np.savetxt(path + "label2.csv", 2*np.ones((n, 1)), fmt='%d', delimiter=",")
    ax = Axes3D(plt.figure())
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    swissroll()
    plane()
