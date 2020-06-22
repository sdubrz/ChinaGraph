import numpy as np
import matplotlib.pyplot as plt


# 当类数特别多时的画法
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
shapes = ['d', '+', '^', 'o', 'p', 's', '*', 'x']


def multi_class_scatter(Y, label, title_str=None):
    n_class = np.max(label)
    n_c = len(colors)
    n_shape = len(shapes)
    if n_class > n_c * n_shape:
        print("类别过多，无法支持")

    (n, m) = Y.shape
    for i in range(0, n):
        current_color = colors[label[i] % n_c]
        current_shape = shapes[label[i] // n_c]
        plt.scatter(Y[i, 0], Y[i, 1], c=current_color, marker=current_shape, alpha=0.7)

    ax = plt.gca()
    ax.set_aspect(1)
    if not (title_str is None):
        plt.title(title_str)
    plt.show()
