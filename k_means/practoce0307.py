#-*-coding:utf-8-*-
# @Time    : 2018/3/7 下午9:52
# @Author  : morening
# @File    : practoce0307.py
# @Software: PyCharm


# http://blog.csdn.net/zouxy09/article/details/17589329
# K-Means
# practoce0307

import numpy as np, matplotlib.pyplot as plt


def load_dataset(path):
    datas = np.loadtxt(path, float)
    return datas


def create_center(K, datas):
    M, N = np.shape(datas)
    center = []
    for k in range(K):
        cent = []
        for n in range(N):
            min = np.min(datas[:, n])
            max = np.max(datas[:, n])
            cent.append(min + np.random.rand() * max)
        center.append(cent)
    return center


def calc_sse(data, center):
    error = np.abs(data - center)
    return np.sum(error)


def calc_center(datas, means_result, K):
    M, N = np.shape(datas)
    center = []
    for k in range(K):
        indexs = [index for index in range(M) if means_result[index][0] == k]
        cent = []
        for n in range(N):
            sum = [datas[index][n] for index in indexs]
            cent.append(np.sum(sum) / len(indexs))
        center.append(cent)

    return center


def kMeans(datas, K):
    M, N = np.shape(datas)
    center = create_center(K, datas)
    cluster = []
    for m in range(M):
        cluster.append([0, 0])

    isChanged = True
    while (isChanged):
        isChanged = False

        for m in range(M):
            data = datas[m]
            index = 0
            min_sse = 9999999
            for k in range(K):
                sse = calc_sse(data, center[k])
                if min_sse > sse:
                    min_sse = sse
                    index = k
            if cluster[m][0] != index or cluster[m][1] != min_sse:
                cluster[m][0] = index
                cluster[m][1] = min_sse
                isChanged = True

        center = calc_center(datas, cluster, K)

    return cluster


def bin_kMeans(datas, K):
    M, N = np.shape(datas)
    cluster = kMeans(datas, 2)
    nCluster = 2
    while nCluster < K:
        index_cluster = 0
        max_sse_cluster = -9999999
        for index_c in range(nCluster):
            sse = np.sum([cluster[index][1] for index in range(M) if cluster[index][0] == index_c])
            if max_sse_cluster < sse:
                max_sse_cluster = sse
                index_cluster = index_c

        means_datas = np.array([datas[index] for index in range(M) if cluster[index][0] == index_cluster])
        means_indexs = [index for index in range(M) if cluster[index][0] == index_cluster]
        means_cluster = kMeans(means_datas, 2)
        means_M, means_N = np.shape(means_datas)
        for means_m in range(means_M):
            if means_cluster[means_m][0] == index_cluster:
                cluster[means_indexs[means_m]][0] = index_cluster
            else:
                cluster[means_indexs[means_m]][0] = nCluster
            cluster[means_indexs[means_m]][1] = means_cluster[means_m][1]

        nCluster += 1

    return cluster


def plot_graph(datas, cluster, K):
    M, N = np.shape(datas)
    color = ['r', 'b', 'g', 'c']
    colors = [color[cluster[m][0]] for m in range(M)]
    plt.scatter(datas[:, 0], datas[:, 1], c=colors)
    center = np.array(calc_center(datas, cluster, K))
    plt.scatter(center[:, 0], center[:, 1], marker='+')
    plt.show()


def show_cluster_sse(cluster):
    sse = np.sum(cluster[:][1])
    print("SSE: %f" % sse)


path = '../data/k_means/testSet.txt'
datas = load_dataset(path)
# cluster = kMeans(datas, 4)
cluster = bin_kMeans(datas, 4)
show_cluster_sse(cluster)
plot_graph(datas, cluster, 4)

