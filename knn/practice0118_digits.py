#-*-coding:utf-8-*-
# @Time    : 2018/1/18 下午10:30
# @Author  : morening
# @File    : practice0118_digits.py
# @Software: PyCharm

import os, numpy as np

def load_txt(path, file_name, SIZE):
    digit = int(file_name.split('_')[0])
    file_path = path + '/' + file_name
    f = open(file_path)
    lines = f.readlines()
    L = []
    for line in lines:
        for k in range(SIZE):
            L.append(int(line[k]))
    L.append(digit)
    return np.array(L)

def load_data(path, SIZE):
    data = []
    files = os.listdir(path)
    for file in files:
        arr = load_txt(path, file, SIZE)
        data.append(arr)

    return np.array(data), files


def compute_dis(train_X, test_X):
    distance = train_X - test_X
    power_dis = np.power(distance, 2)
    return np.sqrt(np.sum(power_dis))


def classify(train_data, test, K):
    L = []
    m, n = np.shape(train_data)
    for k in range(m):
        train = train_data[k, :]
        train_X = train[0: n - 1]
        train_Y = train[n - 1]
        test_X = test[0: n - 1]
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
    max_cnt = 0
    while start < K:
        while start < K and K_L[start] == type:
            start += 1
        if start < K:
            type = K_L[start]
        if max_cnt < start - end:
            max_cnt = start - end
            ret = K_L[start - 1]
        end = start

    return ret


def classify_data(train_data, test_data, K):
    m, n = np.shape(test_data)
    result = []
    for k in range(m):
        result.append(classify(train_data, test_data[k, :], K))
    return result


def populate_result(test_data, test_result):
    m, n = np.shape(test_data)
    cnt = 0
    for data, result in zip(test_data, test_result):
        if data[n-1] == result:
            cnt += 1

    return cnt/m


def show_result(test_files, test_data, test_result, K):
    print('# K = %d' % K)
    print('# 准确度：%s%%' % str(populate_result(test_data, test_result) * 100))
    print('# 测试结果：')
    for file, result in zip(test_files, test_result):
        print(file, result)


K = 10
SIZE = 32
train_path = '../data/knn/digits/trainingDigits'
test_path = '../data/knn/digits/testDigits'
train_data, train_files = load_data(train_path, SIZE)
test_data, test_files = load_data(test_path, SIZE)
test_result = classify_data(train_data, test_data, K)
show_result(test_files, test_data, test_result, K)
