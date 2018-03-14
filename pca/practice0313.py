#-*-coding:utf-8-*-
# @Time    : 2018/3/13 下午9:48
# @Author  : morening
# @File    : practice0313.py
# @Software: PyCharm

# http://blog.jobbole.com/109015/

import numpy as np

def load_dataset():
    datas = [[2.5, 2.4],
             [0.5, 0.7],
             [2.2, 2.9],
             [1.9, 2.2],
             [3.1, 3.0],
             [2.3, 2.7],
             [2, 1.6],
             [1, 1.1],
             [1.5, 1.6],
             [1.1, 0.9]]
    return datas

# 1.分别求各特征的平均值，然后原始数据减去各特征的平均值
# 2.求特征协方差
# 3.求特征协方差的特征值和特征向量
# 4.从大到小排序特征值，选择最大的K个，然后将K个特征向量分别作为列向量组成特征向量矩阵
# 5.将样本点投影到选取的特征向量上，得到降维后的数据

# 如果将矩阵看做运动，那么特征值相当于运动的速度，特征向量相当于运动的方向
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

datas = load_dataset()
datamat = np.mat(datas)
lowmat, recconmat = pca(datamat, 1)
print("降维后的数据：")
print(lowmat)
print("重构后的数据：")
print(recconmat)
