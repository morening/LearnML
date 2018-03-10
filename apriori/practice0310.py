#-*-coding:utf-8-*-
# @Time    : 2018/3/10 上午11:01
# @Author  : morening
# @File    : practice0310.py
# @Software: PyCharm


# 数据来自 http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
# 数据文件 ../data/apriori/congressional_voting_records_data_set.txt

# Attribute Information:
#  1. Class Name: 2 (democrat, republican)
#  2. handicapped-infants: 2 (y,n)
#  3. water-project-cost-sharing: 2 (y,n)
#  4. adoption-of-the-budget-resolution: 2 (y,n)
#  5. physician-fee-freeze: 2 (y,n)
#  6. el-salvador-aid: 2 (y,n)
#  7. religious-groups-in-schools: 2 (y,n)
#  8. anti-satellite-test-ban: 2 (y,n)
#  9. aid-to-nicaraguan-contras: 2 (y,n)
# 10. mx-missile: 2 (y,n)
# 11. immigration: 2 (y,n)
# 12. synfuels-corporation-cutback: 2 (y,n)
# 13. education-spending: 2 (y,n)
# 14. superfund-right-to-sue: 2 (y,n)
# 15. crime: 2 (y,n)
# 16. duty-free-exports: 2 (y,n)
# 17. export-administration-act-south-africa: 2 (y,n)

import numpy as np

def load_dataset(path):
    dataset = np.loadtxt(path, str)
    return dataset

# republican : '0'
# democrat : '1'
# bill(k)'y' : '2k'
# bill(k)'n' : '2k+1'
def convert_datas(dataset):
    datas = []
    for ds in dataset:
        votes = ds.split(',')
        data = {index for index in range(2*len(votes))}
        for index in range(len(votes)):
            if votes[index] == 'republican':
                data.remove(1)
            elif votes[index] == 'democrat':
                data.remove(0)
            elif votes[index] == 'n':
                data.remove(2*index+1)
            elif votes[index] == 'y':
                data.remove(2*index)
            else:
                data.remove(2*index)
                data.remove(2*index+1)
        datas.append(data)

    return datas


def get_base_elements(datas):
    base_elements = []
    for data in datas:
        for itr in data:
            if {itr} not in base_elements:
                base_elements.append({itr})
    return base_elements


def calc_support_rate(element, datas):
    cnt = 0
    for data in datas:
        if element.issubset(data):
            cnt += 1

    return float(cnt) / len(datas)


def generate_mixed_elements(elements):
    mixed_elements = []
    for i_element in elements:
        for j_element in elements:
            if i_element != j_element:
                element = i_element.union(j_element)
                if len(element) == len(i_element) + 1 and element not in mixed_elements:
                    mixed_elements.append(element)

    return mixed_elements


def calc_support(datas, min_support):
    support_list = []
    support_datas = {}
    elements = get_base_elements(datas)
    while (len(elements) > 0):
        new_element = []
        for element in elements:
            rate = calc_support_rate(element, datas)
            if rate >= min_support:
                support_list.append(element)
                support_datas[str(element)] = rate
                new_element.append(element)
        elements = generate_mixed_elements(new_element)

    return support_list, support_datas


def calc_confidence(left_element, right_element, support_datas):
    element1 = left_element.union(right_element)
    element2 = right_element.union(left_element)
    if str(element1) in support_datas:
        return support_datas[str(element1)] / support_datas[str(left_element)]
    elif str(element2) in support_datas:
        return support_datas[str(element2)] / support_datas[str(left_element)]
    else:
        return 0


def remove_duplicate(relation_rules):
    rules = []
    for rule1 in relation_rules:
        for rule2 in relation_rules:
            if rule1 != rule2 and rule1[0].issubset(rule2[0]) and rule1[1] == rule2[1] and rule2 not in rules:
                rules.append(rule2)

    for rule in rules:
        relation_rules.remove(rule)

    return relation_rules


def make_relation_rules(support_list, support_datas, min_confidence):
    relation_rules = []
    for element in support_list:
        if len(element) > 1:
            for e in element:
                left_element = element.copy()
                left_element.remove(e)
                right_element = {e}
                confidence = calc_confidence(left_element, right_element, support_datas)
                if confidence >= min_confidence:
                    relation_rules.append([left_element, right_element, confidence])

    isChanged = True
    while isChanged:
        isChanged = False
        for rule in relation_rules:
            if len(rule[0]) > 1:
                element = rule[0].copy()
                for e in element:
                    left_element = rule[0].copy()
                    left_element.remove(e)
                    right_element = rule[1].copy()
                    right_element.add(e)
                    confidence = calc_confidence(left_element, right_element, support_datas)
                    if confidence >= min_confidence:
                        if [left_element, right_element, confidence] not in relation_rules:
                            relation_rules.append([left_element, right_element, confidence])
                            isChanged = True

    rules = remove_duplicate(relation_rules)
    return rules


def show_supports(support_datas):
    print("频繁项集：")
    print(support_datas)

def show_rules(rules):
    print("关联规则：")
    for rule in rules:
        print("%s => %s conf: %f" % (str(rule[0]), str(rule[1]), rule[2]))

min_support = 0.5
min_confidence = 0.7
print("最小支持度：%.2f，最小可信度：%.2f" % (min_support, min_confidence))
path = '../data/apriori/congressional_voting_records_data_set.txt'
dataset = load_dataset(path)
datas = convert_datas(dataset)
support_list, support_datas = calc_support(datas, min_support)
show_supports(support_datas)
rules = make_relation_rules(support_list, support_datas, min_confidence)
show_rules(rules)