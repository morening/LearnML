#-*-coding:utf-8-*-
# @Time    : 2018/1/12 下午10:30
# @Author  : morening
# @File    : practice0112.py
# @Software: PyCharm

#y = 2*x1 + x2 + 3

import numpy as np, matplotlib.pyplot as plt, time

rate = 0.001
x_train = np.array([[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]])
y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
x_test = np.array([[1, 4], [2, 2], [2, 5], [5, 3], [1, 5], [4, 1]])
len_train = min(len(x_train), len(y_train))

a = 0
b = 0
c = 0
def h(x):
    return  a*x[0] + b*x[1] + c

sum_a = 0
sum_b = 0
sum_c = 0
cnt = 0
before = time.time()

while True:
    cnt += 1
    for x, y in zip(x_train, y_train):
        sum_a += rate * (y - h(x)) * x[0]
        sum_b += rate * (y - h(x)) * x[1]
        sum_c += rate * (y - h(x))
    a = sum_a
    b = sum_b
    c = sum_c
    # plt.plot([h(k) for k in x_test])

    error = 0
    for x, y in zip(x_train, y_train):
        error += (h(x) - y)**2
    error = error / len_train

    if error < 0.000001:
        break

after = time.time()
print('计算轮次：%d，耗时：%f，最小误差：%f' % (cnt, (after-before), error))
print('a = %f' % a)
print('b = %f' % b)
print('c = %f' % c)
print('#测试：')
result = [h(x) for x in x_train]
print(result)
print('#计算：')
result = [h(x) for x in x_test]
print(result)

# plt.show()