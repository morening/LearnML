#-*-coding:utf-8-*-
# @Time    : 2018/1/14 下午9:19
# @Author  : morening
# @File    : practice0114.py
# @Software: PyCharm

# z = theta0 + theta1*x1 + theta2*x2

import numpy as np, matplotlib.pyplot as plt, pylab

def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    X = np.array(data[:, 0:2])
    Y = np.array(data[:, 2])
    return X, Y

def make_feature(x1, x2, degree):
    L = []
    for i in range(degree+1):
        for j in range(degree+1-i):
            L.append((x1**i) * (x2**j))
    return L

def make_polynomial_feature(X1, X2, degree):
    feature = []
    m, n = np.shape(X)
    for k in range(m):
        feature.append(make_feature(X1[k], X2[k], degree))
    return feature

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def g(X, theta):
    return X * theta

def cost_function(X, Y, theta):
    h = sigmoid(X*theta)
    return h - Y

def populate_theta(X, Y, theta):
    g_result = g(np.matrix(X), np.matrix(theta))
    m, n = np.shape(g_result)
    right_cnt = 0
    for result, standard in zip(g_result, Y):
        if result > 0 and standard == 1:
            right_cnt += 1
        elif result <= 0 and standard == 0:
            right_cnt += 1
    return right_cnt/m

def update_theta(X, Y, accuracy):
    rate = 0.001
    m, n = np.shape(X)
    theta = np.ones((n, 1))
    cnt = 0
    while True:
        cnt += 1
        error = cost_function(np.matrix(X), np.matrix(Y).transpose(), np.matrix(theta))
        temp = error.transpose() * np.matrix(X)
        theta -= rate*temp.transpose()

        if populate_theta(X, Y, theta) >= accuracy:
            break
    return  theta, cnt

def train_theta(X, Y, accuracy):
    theta = update_theta(X, Y, accuracy)

    return theta

def show_contour(X, Y, theta, degree):
    pos = pylab.where(Y == 1)
    neg = pylab.where(Y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], c='b', marker='o')
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='x')

    size = 100
    m, n = np.shape(X)
    contour_x1, contour_x2 = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), size), np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), size))
    contour_y = []
    for i in range(size):
        row = []
        for j in range(size):
            feature = make_feature(contour_x1[i, j], contour_x2[i, j], degree)
            row.append(np.dot(feature, theta)[0])
        contour_y.append(row)

    plt.contour(contour_x1, contour_x2, contour_y)
    plt.show()

def compute_result(X, theta):
    g_result = g(np.matrix(X), np.matrix(theta))
    L = []
    for result in g_result:
        if result > 0:
            L.append(1)
        else:
            L.append(0)
    return L

def show_final_result(feature, X, Y, theta, cnt, y, degree):
    accuracy = populate_theta(feature, Y, theta)
    print('#正确率：%s%%' % str(accuracy*100))
    print('#拟合轮次：%d' % cnt)
    print('#拟合参数：')
    print(theta)
    print('#测试结果')
    print(y)
    show_contour(X, Y, theta, degree)



################################################################

degree = 4
accuracy = 0.85
path = '../data/data2.txt'
X, Y = load_data(path)
feature = make_polynomial_feature(X[:, 0], X[:, 1], degree)
theta, cnt = train_theta(feature, Y, accuracy)
y = compute_result(feature, theta)
show_final_result(feature, X, Y, theta, cnt, y, degree)