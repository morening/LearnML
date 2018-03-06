#-*-coding:utf-8-*-
# @Time    : 2018/3/6 下午9:55
# @Author  : morening
# @File    : practice0306.py
# @Software: PyCharm

import numpy as np

def load_dataset(path):
    dataset = np.loadtxt(path, str)
    M, N = np.shape(dataset)
    labels = dataset[0]
    datas = dataset[1:M]
    return labels, datas

def train_naive_bayes(labels, datas):
    M, N = np.shape(datas)
    all_counts = {}
    cate_counts = {}
    for data in datas:
        if str(labels[N-1]+data[N-1]) not in all_counts:
            all_counts[str(labels[N-1]+data[N-1])] = {}
        for k in range(N-1):
            if str(labels[k]+data[k]) not in all_counts[str(labels[N-1]+data[N-1])]:
                all_counts[str(labels[N-1]+data[N-1])][str(labels[k]+data[k])] = 0
            all_counts[str(labels[N-1] + data[N-1])][str(labels[k]+data[k])] += 1

        for k in range(N):
            if str(labels[k]+data[k]) not in cate_counts:
                cate_counts[str(labels[k]+data[k])] = 0
            cate_counts[str(labels[k]+data[k])] += 1

    all_props = {}
    for key1 in all_counts:
        if key1 not in all_props:
            all_props[key1] = {}
        for key2 in all_counts[key1]:
            all_props[key1][key2] = all_counts[key1][key2]/cate_counts[key1]

    cate_props = {}
    for key in cate_counts:
        cate_props[key] = cate_counts[key] / M

    return all_props, cate_props

# p(A|B) = p(B|A)p(A)/p(B)
# 在下面的计算过程中，由于p0和p1的p(B)均相同，故省略，没有参与计算
def test_naive_bayes(all_props, cate_props, labels, datas):
    M, N = np.shape(datas)
    count = 0
    for data in datas:
        p0 = 1
        p1 = 1
        for k in range(N-1):
            if str(labels[k]+data[k]) not in all_props[str(labels[N-1]+'no')]:
                p0 = 0
            elif all_props[str(labels[N-1]+'no')][str(labels[k]+data[k])] > 0:
                p0 *= all_props[str(labels[N-1]+'no')][str(labels[k]+data[k])]

            if str(labels[k]+data[k]) not in all_props[str(labels[N-1]+'yes')]:
                p1 = 0
            elif all_props[str(labels[N-1]+'yes')][str(labels[k]+data[k])] > 0:
                p1 *= all_props[str(labels[N-1]+'yes')][str(labels[k]+data[k])]
        p0 *= cate_props[str(labels[N-1]+'no')]
        p1 *= cate_props[str(labels[N-1]+'yes')]

        if (p0 > p1 and data[N-1] == 'no') or (p1 > p0 and data[N-1] == 'yes'):
            count += 1
            print(str(data)+'=> right')
        else:
            print(str(data)+' => wrong')

    print("正确率：%.2f %%" % (count/M*100))

path = '../data/decision_tree/weather_and_play.txt'
labels, datas = load_dataset(path)
all_props, cate_props = train_naive_bayes(labels, datas)
test_naive_bayes(all_props, cate_props, labels, datas)