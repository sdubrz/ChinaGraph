import numpy as np
import matplotlib.pyplot as plt
from MyDR.LocalPCADR import LocalPCADR
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# 程序循环运行的示例


def run_loop():
    """
    循环版的运行
    :return:
    """
    path = "E:\\ChinaGraph\\Data\\"
    data_name = "coilObjCenterBack10[2]"  # fashionCenter
    path = path + data_name + "\\"
    X0 = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, m) = X0.shape

    # 如果是三维的，则画出三维散点图
    if m == 3:
        ax3d = Axes3D(plt.figure())
        ax3d.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c=label)
        plt.title('original data')
        plt.show()

    if m > 64:
        print("原数据维度过高，现降维至 50 维")
        pca = PCA(n_components=50)
        X = pca.fit_transform(X0)
    else:
        X = X0

    params = {}
    params['neighborhood_type'] = 'knn'  # 'knn' or 'rnn' or 'iter'
    params['n_neighbors'] = 10  # Only used when neighborhood_type is 'knn'
    params['neighborhood_size'] = 0.2  # Only used when neighborhood_type is 'rnn'
    params['alpha'] = 0.9  # the weight of euclidean distance
    params['beta'] = 1.0 - params['alpha']  # the weight of local PCA
    params['distance_type'] = 'spectralNorm'  # 'spectralNorm' or 'mahalanobis'
    params['manifold_dimension'] = 2  # the real dimension of manifolds
    params['perplexity'] = 30.0708  # perplexity in t-SNE
    params['MAX_Distance_iter'] = 10  # max iter of distance computing
    params['use_skeleton'] = False  # boolean value. Whether use skeleton method.
    params['save_path'] = None  # path

    affinity = 'Q'  # affinity 的取值可以为 'cov'  'expCov'  'Q'  'expQ'  'MDS'  't-SNE'  'PCA'  'Isomap'  'LLE'
    # 'geo-t-SNE'  'cTSNE
    frame_work = 't-SNE+'  # frame 的取值可以为 'MDS'  't-SNE'  't-SNE+'
    import time

    # 需要循环的参数
    n_neighbors_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30]
    alpha_list = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35]
    dimension_list = [2, 3, 1, 4, 5, 6, 7, 8]

    loop_count = 0
    for manifold_dimension in dimension_list:
        for n_neighbors in n_neighbors_list:
            for alpha in alpha_list:
                plt.figure(figsize=(20, 20))  # 指定输出文件大小
                loop_count += 1
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(loop_count)
                params['manifold_dimension'] = manifold_dimension
                params['n_neighbors'] = n_neighbors
                params['alpha'] = alpha
                params['beta'] = 1.0 - alpha

                start = time.time()
                dr = LocalPCADR(n_components=2, affinity=affinity, parameters=params, frame=frame_work)

                Y = dr.fit_transform(X)
                finish = time.time()
                print("总共用时:", finish - start)
                run_str = ''  # 用于存放结果的文件名

                # 经典降维方法的文件路径
                classic_methods = ['PCA', 'MDS', 't-SNE', 'Isomap', 'LLE', 'geo-t-SNE']
                if affinity in classic_methods:
                    title_str = affinity
                    if affinity == 't-SNE':
                        title_str = title_str + " perplexity=" + str(params['perplexity'])
                    elif affinity == 'Isomap' or affinity == 'LLE':
                        title_str = title_str + ' n_neighbors=' + str(params['n_neighbors'])
                    elif affinity == 'geo-t-SNE':
                        title_str = title_str + 'n_neighbors=' + str(params['n_neighbors']) + ' perplexity=' + str(
                            params['perplexity'])
                    run_str = title_str
                else:
                    # 我们的降维方法的文件路径
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
                plt.close()


if __name__ == '__main__':
    run_loop()
