#-*-coding:utf-8-*-
# @Time    : 2018/1/15 下午9:00
# @Author  : morening
# @File    : practice0115.py
# @Software: PyCharm

# Fit linear regression with matrix

#y = theta0+theta1*x1+theta2*x2

import numpy as np

def load_data():
    X = np.array([[1, 1, 2], [1, 2, 1], [1, 2, 3], [1, 3, 5], [1, 1, 3], [1, 4, 2], [1, 7, 3], [1, 4, 5], [1, 11, 3], [1, 8, 7]])
    Y = np.array([[7], [8], [10], [14], [8], [13], [20], [16], [28], [26]])
    return X, Y

def h(theta, X):
    y = np.matrix(X) * np.matrix(theta)
    return y

def cost_function(X, Y, theta):
    return h(theta, X) - np.matrix(Y)

def update_theta(X, Y):
    rate = 0.001
    m, n = np.shape(X)
    theta = np.zeros((n, 1))
    cnt = 0
    while True:
        cnt += 1
        error = cost_function(X, Y, theta)
        temp = error.transpose() * np.matrix(X)
        theta -= rate * temp.transpose()
        diff = cost_function(X, Y, theta)
        diff_error = 0
        for d in diff:
            diff_error += abs(d)
        diff_error = diff_error/m
        if diff_error <= 0.00001:
            break

    return theta, cnt

def train_theta():
    X, Y = load_data()
    theta, cnt = update_theta(X, Y)
    return theta, cnt

def test_theta(theta):
    X, Y = load_data()
    y = h(theta, X)
    print('#测试结果：')
    print(y)

def print_result(theta, cnt):
    print('#参数拟合：')
    print(theta)
    print('#轮次：%d' % cnt)

theta, cnt = train_theta()
print_result(theta, cnt)
test_theta(theta)