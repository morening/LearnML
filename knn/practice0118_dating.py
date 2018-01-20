#-*-coding:utf-8-*-
# @Time    : 2018/1/18 下午09:30
# @Author  : morening
# @File    : practice0118_dating.py
# @Software: PyCharm

import numpy as np, matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

def load_data(path, fixed_test_data, test_data_size):
    data = np.loadtxt(path, delimiter=',')
    m, n = np.shape(data)

    if fixed_test_data:
        train_data = data[0:m - test_data_size, :]
        test_data = data[m - test_data_size:m, :]
    else:
        test_L = []
        for k in range(test_data_size):
            index = int(m*np.random.rand())
            while test_L.__contains__(index):
                index = int(m*np.random.rand())
            test_L.append(index)
        train_L = []
        for index in range(m):
            if not test_L.__contains__(index):
                train_L.append(index)

        train_data = data[np.sort(train_L), :]
        test_data = data[np.sort(test_L), :]

    return train_data, test_data


def compute_normalize(num, min, max):
    return (num-min)/(max-min)

def normalize_data(normal_data):
    m, n = np.shape(normal_data)
    for j in range(n-1):
        min = np.min(normal_data[:, j])
        max = np.max(normal_data[:, j])
        for i in range(m):
            normal_data[i, j] = compute_normalize(normal_data[i, j], min, max)

    return normal_data


def normalize(train_data, test_data):
    norm_train_data = normalize_data(np.array(train_data).copy())
    norm_test_data = normalize_data(np.array(test_data).copy())

    return norm_train_data, norm_test_data


def compute_dis(train_X, test_X):
    distance = train_X - test_X
    power_dis = np.power(distance, 2)
    return np.sqrt(np.sum(power_dis))


def classify(train_data, test, K):
    L = []
    m, n = np.shape(train_data)
    for k in range(m):
        train = train_data[k, :]
        train_X = train[0: n-1]
        train_Y = train[n-1]
        test_X = test[0: n-1]
        dis = compute_dis(train_X, test_X)
        L.append([dis, train_Y])

    L.sort()
    L = np.array(L)
    K_L = L[0:K, 1]
    K_L.sort()

    ret = 0
    end = 0
    start = 0
    type = K_L[0]
    max_cnt = 0;
    while start < K:
        while start < K and K_L[start] == type:
            start += 1
        if start < K:
            type = K_L[start]
        if max_cnt < start - end:
            max_cnt = start - end
            ret = K_L[start-1]
        end = start

    return ret


def classify_data(train_data, test_data, K):
    result_L = []
    m, n = np.shape(test_data)
    for k in range(m):
        result_L.append(classify(train_data, test_data[k, :], K))

    return result_L

def populate_result(test_data, test_result):
    m, n = np.shape(test_data)
    cnt = 0
    for data, result in zip(test_data, test_result):
        if data[n-1] == result:
            cnt += 1

    return cnt/m


def show_plot(train_data, test_data, test_result):
    m, n = np.shape(train_data)
    train_data_X = train_data[:, 0:n-1]
    train_data_Y = train_data[:, n-1]
    train_data_X_1 = np.where(train_data_Y == 1)
    train_data_X_2 = np.where(train_data_Y == 2)
    train_data_X_3 = np.where(train_data_Y == 3)

    test_result_1 = np.where(test_result == 1)
    test_result_2 = np.where(test_result == 2)
    test_result_3 = np.where(test_result == 3)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(train_data_X[train_data_X_1, 0], train_data_X[train_data_X_1, 1], train_data_X[train_data_X_1, 2], c='r', marker='o')
    ax.scatter(train_data_X[train_data_X_2, 0], train_data_X[train_data_X_2, 1], train_data_X[train_data_X_2, 2], c='b', marker='o')
    ax.scatter(train_data_X[train_data_X_3, 0], train_data_X[train_data_X_3, 1], train_data_X[train_data_X_3, 2], c='g', marker='o')

    ax.scatter(test_data[test_result_1, 0], test_data[test_result_1, 1], test_data[test_result_1, 2], c='r', marker='x')
    ax.scatter(test_data[test_result_2, 0], test_data[test_result_2, 1], test_data[test_result_2, 2], c='b', marker='x')
    ax.scatter(test_data[test_result_3, 0], test_data[test_result_3, 1], test_data[test_result_3, 2], c='g', marker='x')

    font = FontProperties(fname=r"/System/Library/Fonts/STHeiti Light.ttc", size=14)
    ax.set_xlabel(u'每年获得的飞行常客里程', fontproperties=font)
    ax.set_ylabel(u'玩视频游戏所耗时间百分比', fontproperties=font)
    ax.set_zlabel(u'每周消费的冰淇淋升数', fontproperties=font)

    plt.show()


def show_result(test_data, test_result, K):
    print('# K = %d' % K)
    print('# 准确度：%s%%' % str(populate_result(test_data, test_result)*100))
    print('# 测试结果：')
    for data, result in zip(test_data, test_result):
        print(data, result)


K = 10
test_data_size = 100
fixed_test_data = True     # fixed test data: test with the last 'test_data_size' datas, otherwise select test data randomly
path = '../data/knn/datingTestSet2.txt'
train_data, test_data = load_data(path, fixed_test_data, test_data_size)
norm_train_data, norm_test_data = normalize(train_data, test_data)
test_result = classify_data(norm_train_data, norm_test_data, K)
show_result(test_data, test_result, K)
show_plot(train_data, test_data, np.array(test_result))
