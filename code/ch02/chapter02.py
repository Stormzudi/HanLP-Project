#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Chapter02.py
# @Author: Stormzudi
# @Date  : 2021/1/15 11:12

"""
第二章：词典分词

讲解的内容：
2. 词典分词
2.1 什么是词
2.2 词典
2.3 切分算法
2.4 字典树
2.5 基于字典树的其它算法
2.6 HanLP的词典分词实现

"""


def load_dictionary():
    dic = set()
    # 按行读取字典文件，每行第一个空格之前的字符串提取出来。
    for line in open("CoreNatureDictionary.mini.txt", "r", encoding='utf-8'):
        # 这里只存储每一行中第一个字符串，由于会报错：'gbk' codec can't decode
        # 所有，在encode的时候选择"utf-8"，其中在文本文件中不能存在‘ ’，要换成‘\t’
        dic.add(line[0:line.find('\t')])
    return dic


# (1) 完全切分
# 指的是，找出一段文本中的所有单词。也就是找出了一段话中在dic中的所有词
def fully_segment(text, dic):
    word_list = []
    for i in range(len(text)):  # i 从 0 到text的最后一个字的下标遍历
        for j in range(i + 1, len(text) + 1):  # j 遍历[i + 1, len(text)]区间
            word = text[i:j]  # 取出连续区间[i, j]对应的字符串
            if word in dic:  # 如果在词典中，则认为是一个词
                word_list.append(word)
    return word_list


dic = load_dictionary()
print(fully_segment('就读北京工业大学', dic), '\n')
# 输出了所有可能的单词。由于词库中含有单字，所以结果中也出现了一些单字。



# (2) 正向最长匹配
# 具体来说，就是在以某个下标为起点递增查词的过程中，优先输出更长的单词，这种规则被称为最长匹配算法。
def forward_segment(text, dic):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]                      # 当前扫描位置的单字
        for j in range(i+1, len(text)+1):           # 所有可能的结尾
            word = text[i:j]                        # 从当前位置到结尾的连续字符串
            if word in dic:                         # 判断是否在词典中
                if len(word) > len(longest_word):   # 判断是否能够长度变得更长
                    longest_word = word             # 输出最长词
        word_list.append(longest_word)              # 正向扫描
        i += len(longest_word)
    return word_list

print(forward_segment('就读北京工业大学', dic))
print(forward_segment('研究生命起源', dic))
print(forward_segment('项目的研究', dic), '\n')

# 输出：['就读', '北京', '工业', '大学']
# 输出：['研究生', '命', '起源']


# (3) 逆向最长匹配
def backward_segment(text, dic):
    word_list = []
    i = len(text) - 1
    while i >= 0:
        longest_word = text[i]  # 当前扫描位置的单字
        for j in range(0, i):  # 所有可能的结尾
            word = text[j:i+1]  # 从当前位置到结尾的连续字符串
            if word in dic:  # 判断是否在词典中
                if len(word) > len(longest_word):  # 判断是否能够长度变得更长
                    longest_word = word  # 输出最长词
                    break
        word_list.insert(0, longest_word)
        i -= len(longest_word)
    return word_list


print(backward_segment('就读北京工业大学', dic))
print(backward_segment('研究生命起源', dic))
print(backward_segment('项目的研究', dic), '\n')
# 输出：['项', '目的', '研究']


# (4) 双向最长匹配
"""
这是一种融合两种匹配方法的复杂规则集，流程如下：
(1) 同时执行正向和逆向最长匹配，若两者的词数不同，则返回词数更少的那一个。
(2) 否则，返回两者中单字更少的那一个。当单字数也相同时，优先返回逆向最长匹配的结果。
"""

# 统计单字成词的个数
def count_single_char(word_list: list):
    return sum(1 for word in word_list if len(word) == 1)

# print(count_single_char(['项', '目', '的', '研究']))
def bidirectional_segment(text, dic):
    f = forward_segment(text, dic)
    b = backward_segment(text, dic)
    if len(f) < len(b):
        return f
    elif len(f) > len(b):
        return b
    else:
        if count_single_char(f) < count_single_char(b):
            return f
        else:
            return b

print(bidirectional_segment('研究生命起源', dic))
print(bidirectional_segment('项目的研究', dic), '\n')


# (5) HanLP的词典分词实现
from pyhanlp import *
HanLP.Config.ShowTermNature = False  # 不显示词性

# 可传入自定义字典 [dir1, dir2]
segment = DoubleArrayTrieSegment()
# 激活数字和英文识别
segment.enablePartOfSpeechTagging(True)

print(segment.seg("江西鄱阳湖干枯，中国最大淡水湖变成大草原"))
print(segment.seg("上海市虹口区大连西路550号SISU"))


## 去掉停用词
# 导入文件
def load_from_file(path):
    """
    从词典文件加载DoubleArrayTrie
    :param path: 词典路径
    :return: 双数组trie树
    """
    map = JClass('java.util.TreeMap')()  # 创建TreeMap实例
    with open(path, "r", encoding='utf-8') as src:
        for word in src:
            word = word.strip()  # 去掉Python读入的\n
            map[word] = word
    return JClass('com.hankcs.hanlp.collection.trie.DoubleArrayTrie')(map)


## 去掉停用词
def remove_stopwords_termlist(termlist, trie):
    return [term.word for term in termlist if not trie.containsKey(term.word)]


trie = load_from_file('stopwords.txt')
termlist = segment.seg("江西鄱阳湖干枯了，中国最大的淡水湖变成了大草原")
print('去掉停用词前：', termlist)
print('去掉停用词后：', remove_stopwords_termlist(termlist, trie))