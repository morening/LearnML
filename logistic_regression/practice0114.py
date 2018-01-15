#-*-coding:utf-8-*-
# @Time    : 2018/1/14 下午9:19
# @Author  : morening
# @File    : practice0114.py
# @Software: PyCharm

# z = theta0 + theta1*x1 + theta2*x2

import numpy as np, matplotlib.pyplot as plt, pylab

def load_data():
    data1 = np.loadtxt('../data/data1.txt', delimiter=',')
    X = np.array(data1[:, 0:2])
    m, n = np.shape(X)
    X = np.hstack((np.ones((m, 1)), X))
    Y = np.array(data1[:, 2])
    return X, Y

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def cost_function(X, Y, theta):
    h = sigmoid(X*theta)
    return h - Y

def update_theta(X, Y):
    rate = 0.001
    m, n = np.shape(X)
    theta = np.ones((n, 1))
    for cnt in range(38872):
        error = cost_function(np.matrix(X), np.matrix(Y).transpose(), np.matrix(theta))
        temp = error.transpose() * np.matrix(X)
        theta -= rate*temp.transpose()

    return  theta

def get_min_max(X):
    min = 9999
    max = -9999
    for x in X:
        if min > x:
            min = x
        if max < x:
            max = x

    return min, max

def train_theta():
    X, Y = load_data()
    theta = update_theta(X, Y)
    print('#参数拟合：')
    print(theta)

    return theta

def show_plot(X, Y, theta):
    pos = pylab.where(Y == 1)
    neg = pylab.where(Y == 0)
    plt.scatter(X[pos, 1], X[pos, 2], c='b', marker='o')
    plt.scatter(X[neg, 1], X[neg, 2], c='r', marker='x')
    min, max = get_min_max(X[:, 1])
    plot_x = [min, max]
    plot_y = (-theta[0] - theta[1] * plot_x) / theta[2]
    plt.plot(plot_x, plot_y)
    plt.show()

def compute_result(y, Y):
    m, n = np.shape(y)
    cnt = 0
    for k in range(m):
        if np.abs(y[k]-Y[k]) <= 0.5:
            cnt += 1

    print('#正确率：%s%%' % str(cnt/m*100))

    return cnt/m

def test_theta(theta):
    X, Y = load_data()
    y = sigmoid(np.matrix(X) * theta)
    compute_result(y, Y)
    print('#测试结果：')
    print(y)
    show_plot(X, Y, theta)

theta = train_theta()
test_theta(theta)