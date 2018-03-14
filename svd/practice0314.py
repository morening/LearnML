#-*-coding:utf-8-*-
# @Time    : 2018/3/14 下午9:41
# @Author  : morening
# @File    : practice0314.py
# @Software: PyCharm

import numpy as np, numpy.linalg as lg

# 书中给出的例子是基于物品计算相似度的
# 如果用户数量大于物品数量，则基于物品计算相似度，反之依然

def load_dataset():
    dataset = [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
               [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
               [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
               [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
               [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
               [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
               [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
               [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
    return np.mat(dataset)


def eclud_sim(inA, inB):
    return 1.0 / (1.0 + lg.norm(inA - inB))

def stand_est(datamat, user, sim_func, item):
    M, N = np.shape(datamat)
    sim_total = 0.0
    rate_sim_total = 0.0
    for j in range(N):
        user_rating = datamat[user, j]
        if user_rating == 0:
            continue
        overlap = np.nonzero(np.logical_and(datamat[:, item].A > 0, datamat[:, j].A > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = sim_func(datamat[overlap, item], datamat[overlap, j])
        # print("the %d and %d similarity is: %f" % (item, j, similarity))
        sim_total += similarity
        rate_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rate_sim_total/sim_total

def svd_est(datamat, user, sim_func, item):
    M, N = np.shape(datamat)
    sim_total = 0.0
    rate_sim_total = 0.0
    # 这部分每次计算都是公用的
    U, S, VT = lg.svd(datamat)
    K_90 = 0
    for k in range(N):
        if np.sum(S[:k]) >= np.sum(S)*0.9:
            K_90 = k
            break
    S_k = np.mat(np.eye(K_90) * S[:K_90])
    xformed_items = datamat.T * U[:, :K_90] * S_k.I
    # end ===================
    for j in range(N):
        user_rating = datamat[user, j]
        if user_rating == 0 or j == item:
            continue
        similarity = sim_func(xformed_items[item, :].T, xformed_items[j, :].T)
        # print("the %d and %d similarity is: %f" % (item, j, similarity))
        sim_total += similarity
        rate_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rate_sim_total / sim_total


def recommend(datamat, user, N=3, sim_func=eclud_sim, est_func=svd_est):
    unrated_items = np.nonzero(datamat[user, :].A == 0)[1]
    if len(unrated_items) == 0:
        return 'you rated everything'
    item_scores = []
    for item in unrated_items:
        estimated_score = est_func(datamat, user, sim_func, item)
        item_scores.append((item, estimated_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:N]


datamat = load_dataset()
result_stand = recommend(datamat, 10, est_func=stand_est)
print("标准推荐：")
print(result_stand)
result_svd = recommend(datamat, 10, est_func=svd_est)
print("SVD推荐：")
print(result_svd)
