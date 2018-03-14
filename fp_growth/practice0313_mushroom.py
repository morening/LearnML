#-*-coding:utf-8-*-
# @Time    : 2018/3/13 下午10:19
# @Author  : morening
# @File    : practice0313_mushroom.py
# @Software: PyCharm

# 使用fp-growth发现毒蘑菇特征的频繁项集
# 数据来自 http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/
# 数据文件 ../data/apriori/agaricus-lepiota.data

class Node:

    def __init__(self, value, cnt, parent):
        self.value = value
        self.cnt = cnt
        self.childs = {}
        self.parent = parent
        self.next = None


import numpy as np

def load_dataset(path):
    dataset = np.loadtxt(path, str)
    return dataset

def convert_datas(dataset):
    convert_list = []
    values = dataset[0].split(',')
    N = len(values)
    count = 0
    for index in range(N):
        M = {}
        for ds in dataset:
            values = ds.split(',')
            if values[index] not in M:
                M[values[index]] = count
                count += 1
        convert_list.append(M)

    datas = []
    for ds in dataset:
        values = ds.split(',')
        data = []
        for index in range(len(values)):
            value = convert_list[index][values[index]]
            data.append(str(value))
        datas.append(data)

    return datas

def filter_datas(datas, min_support=1):
    counters = {}
    for data in datas:
        for value in data:
            if value not in counters:
                counters[value] = Node(value, 0, None)
            counters[value].cnt += 1

    filtered_datas = []
    for data in datas:
        filtered_data = []
        for itr in data:
            if counters[itr].cnt >= min_support:
                filtered_data.append(itr)
        filtered_datas.append(filtered_data)

    for filtered_data in filtered_datas:
        filtered_data.sort(key=lambda data: counters[data].cnt, reverse=True)

    removed_counters = []
    for key in counters:
        if counters[key].cnt < min_support:
            removed_counters.append(key)

    for key in removed_counters:
        counters.pop(key)

    return filtered_datas, counters

def create_fp_tree(filtered_datas, counters):
    root = Node('root', 0, None)
    for filtered_data in filtered_datas:
        ptr = root
        for value in filtered_data:
            if value in ptr.childs:
                ptr.childs[value].cnt += 1
            else:
                ptr.childs[value] = Node(value, 1, ptr)
                ptr.childs[value].next = counters[value].next
                counters[value].next = ptr.childs[value]
            ptr = ptr.childs[value]

    return root, counters

def show_fp_tree(root, depth):
    print("  "*depth+root.value+" "+str(root.cnt))
    childs = root.childs
    for value in childs:
        show_fp_tree(childs[value], depth+1)

def find_prefix_datas(head):
    prefix_datas = []
    ptr = head.next
    while ptr != None:
        for k in range(ptr.cnt):
            datas = []
            parent = ptr.parent
            while parent.parent != None:
                datas.append(parent.value)
                parent = parent.parent
            datas.reverse()
            if len(datas) > 0:
                prefix_datas.append(datas)
        ptr = ptr.next

    return prefix_datas

def mine_fp_tree(counters, min_support):
    if len(counters) == 0:
        return None
    fs = []
    for key in counters:
        if len(counters) == 1:
            return [[key]]
        if [key] not in fs:
            fs.append([key])
        prefix_datas = find_prefix_datas(counters[key])
        if len(prefix_datas) == 0:
            continue
        filtered_prefix_datas, prefix_counters = filter_datas(prefix_datas, min_support)
        if len(prefix_counters) == 0:
            continue
        prefix_root, prefix_counters = create_fp_tree(filtered_prefix_datas, prefix_counters)
        # show_fp_tree(prefix_root, 0)
        prefix_fs = mine_fp_tree(prefix_counters, min_support)
        for f in prefix_fs:
            if f not in fs:
                fs.append(f)
        for f in prefix_fs:
            s1 = set([key])
            s2 = set(f)
            s3 = s1.union(s2)
            L = list(s3)
            L.sort()
            if L not in fs:
                fs.append(L)

    return fs

def show_frequency_set(fs):
    print("频繁项集：")
    print(fs)


min_support = 4062 #等同于apriori算法中最小支持度0.5
path = '../data/apriori/agaricus-lepiota.data'
dataset = load_dataset(path)
datas = convert_datas(dataset)
filtered_datas, counters = filter_datas(datas, min_support)
root, counters = create_fp_tree(filtered_datas, counters)
show_fp_tree(root, 0)
fs = mine_fp_tree(counters, min_support)
show_frequency_set(fs)