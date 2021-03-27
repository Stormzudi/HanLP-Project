#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : chapter03.py
# @Author: Stormzudi
# @Date  : 2021/1/15 20:27


"""
第三章： HanLP分词与用户词典的集成
"""

from pyhanlp import *
ViterbiSegment = SafeJClass('com.hankcs.hanlp.seg.Viterbi.ViterbiSegment')

segment = ViterbiSegment()
sentence = "社会摇摆简称社会摇"
segment.enableCustomDictionary(False)
print("不挂载词典：", segment.seg(sentence))

CustomDictionary.insert("社会摇", "nz 100")
segment.enableCustomDictionary(True)
print("低优先级词典：", segment.seg(sentence))
segment.enableCustomDictionaryForcing(True)
print("高优先级词典：", segment.seg(sentence))