#-*-coding:utf-8-*-
# @Time    : 2018/3/1 下午09:01
# @Author  : morening
# @File    : practice0301.py
# @Software: PyCharm

import numpy as np

ID3 = 'ID3'
C45 = 'C4.5'
CART = 'CART'

def convert_to_list(dataset):
    list = []
    for data in dataset:
        temp = []
        for item in data:
            temp.append(item)
        list.append(temp)

    return list

def load_dataset(path):
    dataset = np.loadtxt(path, str)
    dataset = convert_to_list(dataset)
    M, N = np.shape(dataset)
    labels = dataset[0]
    datas = dataset[1:M]

    return labels, datas

def calc_basic_ent(datas, feature):
    M, N = np.shape(datas)
    counts = {}
    for data in datas:
        if data[feature] not in counts:
            counts[data[feature]] = 0
        counts[data[feature]] += 1

    basic_ent = 0
    for key in counts:
         basic_ent -= (counts[key]/M)*np.log2(counts[key]/M)

    return basic_ent

def calc_ent(datas, feature):
    M, N = np.shape(datas)
    counts = {}
    for data in datas:
        if data[feature] not in counts:
            counts[data[feature]] = {}
        if data[N-1] not in counts[data[feature]]:
            counts[data[feature]][data[N-1]] = 0
        counts[data[feature]][data[N - 1]] += 1

    ret = 0
    for key1 in counts:
        num = 0
        for key2 in counts[key1]:
            num += counts[key1][key2]

        ent = 0
        for key2 in counts[key1]:
            ent -= (counts[key1][key2]/num)*np.log2(counts[key1][key2]/num)

        ret += num/M * ent

    return ret

# 增益 = 基础信息熵 - 对某一分裂特征的条件熵
def get_best_feature_by_ID3(datas):
    M, N = np.shape(datas)
    basic_ent = calc_basic_ent(datas, N-1)

    max_gain = -999999
    best_feature = 0
    for feature in range(N-1):
        ent = calc_ent(datas, feature)
        gain = basic_ent - ent
        if gain > max_gain:
            max_gain = gain
            best_feature = feature

    return best_feature

# 增益率 = 增益 / 内在信息
# 内在信息 = 对某一分裂属相的基础信息熵
def get_best_feature_by_C45(datas):
    M, N = np.shape(datas)
    basic_ents = []
    for feature in range(N):
        basic_ents.append(calc_basic_ent(datas, feature))

    max_rate = -99999999
    best_key = 0
    for feature in range(N-1):
        ent = calc_ent(datas, feature)
        gain = basic_ents[N-1] - ent
        rate = gain / basic_ents[feature]
        if max_rate < rate:
            max_rate = rate
            best_key = feature

    return best_key

def get_best_feature(datas, algorithm):
    if algorithm == ID3:
        return get_best_feature_by_ID3(datas)

    return get_best_feature_by_C45(datas)

def split_by_feature(datas, best_feature):
    child_datas = {}
    for data in datas:
        if data[best_feature] not in child_datas:
            child_datas[data[best_feature]] = []
        key = data.pop(best_feature)
        child_datas[key].append(data)

    return child_datas

def copy_deeply(datas):
    new_datas = []
    for data in datas:
        new_datas.append(data.copy())
    return new_datas

def create_decision_tree(labels, datas, algorithm):
    if len(labels) == 0:
        return
    M, N = np.shape(datas)
    if calc_basic_ent(datas, N-1) == 0:
        return datas[0][N-1]
    parent = {}
    best_feature = get_best_feature(datas, algorithm)
    parent['label'] = labels[best_feature]
    parent['child'] = {}

    child_label = labels.copy()
    child_label.pop(best_feature)
    child_datas = copy_deeply(datas)
    splite_datas = split_by_feature(child_datas, best_feature)
    for key in splite_datas:
        parent['child'][key] = create_decision_tree(child_label, splite_datas[key], algorithm)

    return parent

# 特征A条件下的基尼指数 Gini(D|A) = |D1|/D*Gini(D1)+|D2|/D*Gini(D2)
# 样本D的基尼指数 Gini(p) = 2p(1-p)
def get_best_feature_by_CART(datas):
    M, N = np.shape(datas)
    best_feature = 0
    best_splite = 0
    min_gini = 99999
    for feature in range(N-1):
        counts = {}
        for data in datas:
            key1 = data[feature]
            if key1 not in counts:
                counts[key1] = {}
            key2 = data[N-1]
            if key2 not in counts[key1]:
                counts[key1][key2] = 0
            counts[key1][key2] += 1

        for key1 in counts:
            sum_all = 0
            for key2 in counts[key1]:
                sum_all += counts[key1][key2]
            sum_key = 0
            for key3 in counts:
                if key2 in counts[key3]:
                    sum_key += counts[key3][key2]
            gini = (sum_all/M)*2*(counts[key1][key2]/sum_all)*(1-counts[key1][key2]/sum_all) + ((M-sum_all)/M)*2*((sum_key-counts[key1][key2])/(M-sum_all))*(1-((sum_key-counts[key1][key2])/(M-sum_all)))
            if min_gini > gini:
                min_gini = gini
                best_feature = feature
                best_splite = key1

    return best_feature, best_splite

def calc_gini(datas):
    M, N = np.shape(datas)
    counts = {}
    for data in datas:
        if data[N-1] not in counts:
            counts[data[N-1]] = 0
        counts[data[N-1]] += 1

    gini = 2*(counts[datas[0][N-1]]/M)*(1-counts[datas[0][N-1]]/M)
    return gini

def create_decision_tree_by_CART(labels, datas):
    M, N = np.shape(datas)
    if calc_gini(datas) == 0:
        return datas[0][N-1]
    if len(labels) == 1:
        return 'yes/no'
    best_feature, best_splite = get_best_feature_by_CART(datas)
    parent = {}
    parent['label'] = labels[best_feature]
    new_labels = labels.copy()
    new_labels.pop(best_feature)

    new_datas = copy_deeply(datas)
    splite_datas = split_by_feature(new_datas, best_feature)
    new_splite_datas = {best_splite: splite_datas[best_splite]}
    list = []
    for splite in splite_datas:
        if splite != best_splite:
            for data in splite_datas[splite]:
                list.append(data)
    new_splite_datas['others'] = list

    parent['child'] = {}
    for splite in new_splite_datas:
        parent['child'][splite] = create_decision_tree_by_CART(new_labels, new_splite_datas[splite])

    return parent

def print_decision_tree(root, algorithm):
    print(algorithm+"建树：")
    print(root)

path = '../data/decision_tree/weather_and_play.txt'
labels, datas = load_dataset(path)
root = create_decision_tree(labels, datas, ID3)
print_decision_tree(root, ID3)
root = create_decision_tree(labels, datas, C45)
print_decision_tree(root, C45)
root = create_decision_tree_by_CART(labels, datas)
print_decision_tree(root, CART)