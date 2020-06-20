import numpy as np
import matplotlib.pyplot as plt
from MyDR.LocalPCADR import LocalPCADR
from mpl_toolkits.mplot3d import Axes3D


# 程序运行的示例


def run_example():
    """
    一个使用 local PCA 降维方法的示例
    :return:
    """
    path = "E:\\ChinaGraph\\Data\\2plane90\\"
    X = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape

    # 如果是三维的，则画出三维散点图
    if m == 3:
        ax3d = Axes3D(plt.figure())
        ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=label)
        plt.title('original data')
        plt.show()

    params = {}
    params['neighborhood_type'] = 'knn'  # 'knn' or 'rnn' or 'iter'
    params['n_neighbors'] = 10  # Only used when neighborhood_type is 'knn'
    params['neighborhood_size'] = 0.2  # Only used when neighborhood_type is 'rnn'
    params['alpha'] = 0.5  # the weight of euclidean distance
    params['beta'] = 1.0 - params['alpha']  # the weight of local PCA
    params['distance_type'] = 'spectralNorm'  # 'spectralNorm' or 'mahalanobis'
    params['manifold_dimension'] = 2  # the real dimension of manifolds
    params['perplexity'] = 30.01  # perplexity in t-SNE
    params['MAX_Distance_iter'] = 10  # max iter of distance computing
    params['use_skeleton'] = False  # boolean value. Whether use skeleton method.

    affinity = 't-SNE'  # affinity 的取值可以为 'cov'  'expCov'  'Q'  'expQ'  'MDS'  't-SNE'  'PCA'  'Isomap'  'LLE'
    # 'geo-t-SNE'
    frame_work = 't-SNE+'  # frame 的取值可以为 'MDS'  't-SNE'  't-SNE+'
    dr = LocalPCADR(n_components=2, affinity=affinity, parameters=params, frame=frame_work, manifold_dimension=2)

    Y = dr.fit_transform(X)
    run_str = ''  # 用于存放结果的文件名

    # 骨架点结果的画图
    if params['use_skeleton']:
        skeleton_Y = dr.skeleton_Y
        plt.scatter(skeleton_Y[:, 0], skeleton_Y[:, 1])
        plt.title('skeleton points')
        plt.show()

    # 经典降维方法的画图
    classic_methods = ['PCA', 'MDS', 't-SNE', 'Isomap', 'LLE', 'geo-t-SNE']
    if affinity in classic_methods:
        plt.scatter(Y[:, 0], Y[:, 1], c=label)
        ax = plt.gca()
        ax.set_aspect(1)
        title_str = affinity
        if affinity == 't-SNE':
            title_str = title_str + " perplexity=" + str(params['perplexity'])
        elif affinity == 'Isomap' or affinity == 'LLE':
            title_str = title_str + ' n_neighbors=' + str(params['n_neighbors'])
        elif affinity == 'geo-t-SNE':
            title_str = title_str + 'n_neighbors=' + str(params['n_neighbors']) + ' perplexity=' + str(params['perplexity'])
        plt.title(title_str)
        run_str = title_str
    else:
        # 我们的降维方法的画图
        plt.scatter(Y[:, 0], Y[:, 1], c=label)
        ax = plt.gca()
        ax.set_aspect(1)
        title_str = 'Frame[' + frame_work + '] ' + affinity + ' alpha=' + str(params['alpha']) + ' beta=' + str(
            params['beta']) + ' manifold_dimension=' + str(params['manifold_dimension'])
        if params['use_skeleton']:
            title_str = 'skeletonMethod ' + title_str
        if params['neighborhood_type'] == 'knn':
            title_str = title_str + ' k=' + str(params['n_neighbors'])
        elif params['neighborhood_type'] == 'rnn':
            title_str = title_str + ' r=' + str(params['neighborhood_size'])
        if frame_work == 't-SNE' or frame_work == 't-SNE+':
            title_str = title_str + " perplexity=" + str(params['perplexity'])
        if params['neighborhood_type'] == 'iter':
            title_str = title_str + ' distanceIter=' + str(params['MAX_Distance_iter'])
        plt.title(title_str)
        run_str = title_str
    np.savetxt(path + run_str + ".csv", Y, fmt='%.18e', delimiter=",")
    plt.savefig(path + run_str + ".png")
    plt.show()


if __name__ == '__main__':
    run_example()




