#-*-coding:utf-8-*-
# @Time    : 2018/3/9 下午9:47
# @Author  : morening
# @File    : practice0309.py
# @Software: PyCharm


# 《机器学习实战》的说明例子
# Apriori
# practice0309
# 频繁项集：如果一个项集是非频繁集，那么它的所有超集也是非频繁的
# 关联规则：如果某条规则不满足最小可行度要求，那么该规则的所有子集也不满足最小可信度要求

def load_dataset():
    datas = [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
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
    for i in range(len(elements)):
        for j in range(i, len(elements)):
            element = elements[i].union(elements[j])
            if len(element) == len(elements[i]) + 1 and element not in mixed_elements:
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
min_confidence = 0.5
print("最小支持度：%.2f，最小可信度：%.2f" % (min_support, min_confidence))
datas = load_dataset()
support_list, support_datas = calc_support(datas, min_support)
show_supports(support_datas)
rules = make_relation_rules(support_list, support_datas, min_confidence)
show_rules(rules)
