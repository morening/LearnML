#-*-coding:utf-8-*-
# @Time    : 2018/3/10 下午3:56
# @Author  : morening
# @File    : practice0310_mushroom.py
# @Software: PyCharm

# 数据来自 http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/
# 数据文件 ../data/apriori/agaricus-lepiota.data

# Attribute Information: (classes: edible=e, poisonous=p)
#  1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
#                               knobbed=k,sunken=s
#  2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
#  3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
#                               pink=p,purple=u,red=e,white=w,yellow=y
#  4. bruises?:                 bruises=t,no=f
#  5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
#                               musty=m,none=n,pungent=p,spicy=s
#  6. gill-attachment:          attached=a,descending=d,free=f,notched=n
#  7. gill-spacing:             close=c,crowded=w,distant=d
#  8. gill-size:                broad=b,narrow=n
#  9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
#                               green=r,orange=o,pink=p,purple=u,red=e,
#                               white=w,yellow=y
# 10. stalk-shape:              enlarging=e,tapering=t
# 11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
#                               rhizomorphs=z,rooted=r,missing=?
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
#                               pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
#                               pink=p,red=e,white=w,yellow=y
# 16. veil-type:                partial=p,universal=u
# 17. veil-color:               brown=n,orange=o,white=w,yellow=y
# 18. ring-number:              none=n,one=o,two=t
# 19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
#                               none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
#                               orange=o,purple=u,white=w,yellow=y
# 21. population:               abundant=a,clustered=c,numerous=n,
#                               scattered=s,several=v,solitary=y
# 22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
#                               urban=u,waste=w,woods=d


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

    # datas = []
    # for ds in dataset:
    #     values = ds.split(',')
    #     data = {k for k in range(count)}
    #     for index in range(len(values)):
    #         value = convert_list[index][values[index]]
    #         min_value = min(convert_list[index].values())
    #         max_value = max(convert_list[index].values())
    #         for k in range(min_value, (max_value+1)):
    #             if k != value:
    #                 data.remove(k)
    datas = []
    for ds in dataset:
        values = ds.split(',')
        data = []
        for index in range(len(values)):
            value = convert_list[index][values[index]]
            data.append(value)
        datas.append(data)

    return datas


def get_base_elements(datas):
    base_elements = []
    for data in datas:
        for itr in data:
            if [itr] not in base_elements:
                base_elements.append([itr])
    base_elements.sort()
    return base_elements


def calc_support_rate(element, datas):
    cnt = 0
    for data in datas:
        if set(element).issubset(set(data)):
            cnt += 1

    return float(cnt) / len(datas)


def generate_mixed_elements(elements):
    mixed_elements = []
    for i in range(len(elements)):
        for j in range(i, len(elements)):
            S = set(elements[i]).union(set(elements[j]))
            element = list(S)
            element.sort()
            if len(element) == len(elements[j]) + 1 and element not in mixed_elements:
                mixed_elements.append(element)


    # for i_element in elements:
    #     for j_element in elements:
    #         S = set(i_element).union(set(j_element))
    #         element = list(S)
    #         if len(element) == len(i_element) + 1 and element not in mixed_elements:
    #             mixed_elements.append(element)
    mixed_elements.sort()
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
        new_element.sort()
        elements = generate_mixed_elements(new_element)

    return support_list, support_datas


def calc_confidence(left_element, right_element, support_datas):
    element = left_element + right_element
    element.sort()
    if str(element) in support_datas:
        return support_datas[str(element)] / support_datas[str(left_element)]
    else:
        return 0


def remove_duplicate(relation_rules):
    rules = []
    for rule1 in relation_rules:
        for rule2 in relation_rules:
            if rule1 != rule2 and set(rule1[0]).issubset(set(rule2[0])) and rule1[1] == rule2[1] and rule2 not in rules:
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
                right_element = [e]
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
                    right_element.append(e)
                    right_element.sort()
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
path = '../data/apriori/agaricus-lepiota.data'
dataset = load_dataset(path)
datas = convert_datas(dataset)
support_list, support_datas = calc_support(datas, min_support)
show_supports(support_datas)
rules = make_relation_rules(support_list, support_datas, min_confidence)
show_rules(rules)