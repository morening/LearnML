#-*-coding:utf-8-*-
# @Time    : 2018/3/13 下午9:52
# @Author  : morening
# @File    : practice0313_secom.py
# @Software: PyCharm

# 数据来源 http://archive.ics.uci.edu/ml/machine-learning-databases/secom/

import numpy as np, matplotlib.pyplot as plt

def load_dataset(path):
    dataset = np.loadtxt(path, float)
    return dataset

def fix_dataset(dataset):
    M, N = np.shape(dataset)
    for n in range(N):
        sum = 0
        count = 0
        for data in dataset:
            if np.isnan(data[n]) == False and np.nonzero(data[n]):
                sum += data[n]
                count += 1
        mean_val = sum / count

        for data in dataset:
            if np.isnan(data[n]) == True:
                data[n] = mean_val

    return np.mat(dataset)

def show_eig_result(datamat):
    mean_vals = np.mean(datamat, axis=0)
    mean_removed = datamat - mean_vals
    covmat = np.cov(mean_removed, rowvar=False)
    eig_vals, eig_vects = np.linalg.eig(covmat)
    np.sort(eig_vals)
    print("特征值：")
    print(eig_vals)
    eig_sum = np.sum(eig_vals)
    eig_pretage = eig_vals.copy()
    for k in range(1, len(eig_pretage)):
        eig_pretage[k] = eig_pretage[k-1] + eig_pretage[k]
    eig_pretage = eig_pretage/eig_sum
    print("特征值百分比：")
    print([float(eig_pretage[k]) for k in range(20)])

    plt.scatter([k for k in range(21)], [float(eig_pretage[k]) for k in range(21)], marker='x')
    plt.plot([k for k in range(21)], [float(eig_pretage[k]) for k in range(21)])
    plt.show()


def pca(datamat, top_num):
    mean_vals = np.mean(datamat, axis=0)
    mean_removed = datamat - mean_vals
    covmat = np.cov(mean_removed, rowvar=False)
    eig_vals, eig_vects = np.linalg.eig(covmat)
    eig_vals_index = np.argsort(eig_vals)
    eig_vals_index = eig_vals_index[:-(top_num+1):-1]
    recon_eig_vects = eig_vects[:, eig_vals_index]
    lowmat = mean_removed * np.mat(recon_eig_vects)
    recconmat = (lowmat*recon_eig_vects.transpose())+mean_vals
    return lowmat, recconmat


path = '../data/pca/secom.data'
dataset = load_dataset(path)
datamat = fix_dataset(dataset)
# 通过特征值观察数据特征情况
show_eig_result(datamat)
# 确定特征值选取范围，进行主成分分析（对数据启动二向箔打击，哈哈）
# 前20个特征大约包括了99.27%的特征信息
lowmat, recconmat = pca(datamat, 20)
print("降维后的数据：")
print(lowmat)
print("重构后的数据：")
print(recconmat)
