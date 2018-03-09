#-*-coding:utf-8-*-
# @Time    : 2018/3/8 下午9:48
# @Author  : morening
# @File    : practice0308.py
# @Software: PyCharm

# http://blog.csdn.net/guyuealian/article/details/70995333
# Adaboost
# practice0308

import numpy as np, matplotlib.pyplot as plt


class Classifier:
    feature = 0
    split = 0
    tree = 0
    alpha = 0

    def __init__(self, feature, split, tree):
        self.feature = feature
        self.split = split
        self.tree = tree

    def __repr__(self):
        return "feature: {}  split: {}  tree: {}  alpha:{}".format(self.feature, self.split, self.tree, self.alpha)


def load_dataset():
    datas = [[1, 5, 1], [2, 2, 1], [3, 1, -1], [4, 6, -1], [6, 8, 1],
             [6, 5, -1], [7, 9, 1], [8, 7, 1], [9, 8, -1], [10, 2, -1]]
    return np.array(datas)


# 单层决策树
def stump_classify(datas, feature, split, tree):
    M, N = np.shape(datas)
    classified_value = np.zeros((M, 1))
    if tree == 0:
        classified_value[datas[:, feature] < split] = -1
        classified_value[datas[:, feature] >= split] = 1
    else:
        classified_value[datas[:, feature] < split] = 1
        classified_value[datas[:, feature] >= split] = -1
    return classified_value


def build_classifier(datas, D):
    M, N = np.shape(datas)
    feature_N = N - 1
    min_error_rate = np.inf
    min_classified_value = None
    min_split = 0
    min_feature = 0
    min_tree = 0
    for feature in range(feature_N):
        min = np.min(datas[:, feature])
        max = np.max(datas[:, feature])
        step_len = (max - min) / M
        for step in range(M):
            split = min + step * step_len
            for tree in range(2):
                classified_value = stump_classify(datas, feature, split, tree)
                errors = np.ones((M, 1))
                for m in range(M):
                    if classified_value[m] == datas[m, N - 1]:
                        errors[m] = 0
                error_rate = np.sum(errors * D)
                if min_error_rate > error_rate:
                    min_error_rate = error_rate
                    min_classified_value = classified_value
                    min_split = split
                    min_feature = feature
                    min_tree = tree

    min_classifier = Classifier(min_feature, min_split, min_tree)

    return min_classifier, min_error_rate, min_classified_value


def calc_error_rate(datas, classified_value):
    M, N = np.shape(datas)
    error_cnt = 0
    for k in range(M):
        if np.sign(classified_value[k]) != datas[k, N - 1]:
            error_cnt += 1
    return error_cnt / M

# 初始样本权重向量D = [1/M， 1/M，...]
# 构造弱分类器，获得分类结果，根据样本权重向量，求得错误率
# 通过比较错误率，获得最优的弱分类器，记录下其特征值(feature)，分割点(split)和树结构(tree)
# 分类器权重值alpha = 1/2ln((1-error)/error)，增大效果好的弱分类器的权重，减少效果差的权重
# 更新样本权重向量D，对于分类正确的样本：D = D/(2*error)；对于分类错误的样本：D = D/(2*(1-error))
# 在下次迭代中，减少分类正确的样本的权重，增加分类错误的样本的权重
def ada_boost(datas, iterNum):
    final_classifier = []
    M, N = np.shape(datas)
    D = np.ones((M, 1)) / M
    final_classified_value = np.zeros((M, 1))
    for iter in range(iterNum):
        classifier, error, classified_value = build_classifier(datas, D)
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-16))
        classifier.alpha = alpha
        final_classifier.append(classifier)
        for k in range(M):
            if classified_value[k] == datas[k, N - 1]:
                D[k] = D[k] / (2 * (1 - error))
            else:
                D[k] = D[k] / (2 * error)

        final_classified_value += alpha * classified_value

        if calc_error_rate(datas, final_classified_value) == 0.0:
            break

    return final_classifier


def show_final_result(datas, final_classifier):
    M, N = np.shape(datas)
    color = ['c', 'r', 'b']
    colors = [color[datas[index, N - 1]] for index in range(M)]
    plt.scatter(datas[:, 0], datas[:, 1], c=colors)
    for classifier in final_classifier:
        min = np.array([float(np.min(datas[:, 0])), float(np.min(datas[:, 1]))])
        max = np.array([float(np.max(datas[:, 0])), float(np.max(datas[:, 1]))])
        min[classifier.feature] = classifier.split
        max[classifier.feature] = classifier.split
        plt.plot([min[0], max[0]], [min[1], max[1]], c='g')
    plt.show()


def classify(datas, classifier):
    M, N = np.shape(datas)
    feature = classifier.feature
    split = classifier.split
    tree = classifier.tree
    classified_value = np.zeros((M, 1))
    if tree == 0:
        classified_value[datas[:, feature] < split] = -1
        classified_value[datas[:, feature] >= split] = 1
    else:
        classified_value[datas[:, feature] < split] = 1
        classified_value[datas[:, feature] >= split] = -1

    return classified_value


def test_classifier(datas, final_classifier):
    M, N = np.shape(datas)
    final_classified_value = np.zeros((M, 1))
    for classifier in final_classifier:
        classified_value = classify(datas, classifier)
        final_classified_value += classifier.alpha * classified_value

    error_cnt = 0
    for k in range(M):
        if np.sign(final_classified_value[k]) != datas[k, N - 1]:
            error_cnt += 1

    print("错误率：%.2f %%" % (error_cnt / M * 100))
    print(np.sign(final_classified_value.transpose()))


datas = load_dataset()
final_classifier = ada_boost(datas, 10)
print(final_classifier)
test_classifier(datas, final_classifier)
show_final_result(datas, final_classifier)