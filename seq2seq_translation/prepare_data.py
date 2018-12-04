# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: liuqm
# datetime:2018/12/3 下午9:13
# software: PyCharm

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1


# self.word2index = {} 单词到索引
# self.word2count = {} 索引到单词
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 这些文件全部采用Unicode编码,为了简化我们将Unicode字符转换为ASCII, 使所有内容小写,并修剪大部分标点符号
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 小写,修剪和删除非字母字符
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 读取文件
def readLangs(lang1, lang2, reverse=False):
    # print("Reading lines...")

    # 读取文件并按行分开
    # 文件下载路径：https://tatoeba.org/eng/downloads
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 将每一行分成两列并进行标准化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 翻转对,Lang实例化
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# 由于有很多例句,我们希望快速训练,我们会将数据集裁剪为相对简短的句子.
# 这里的单词的最大长度是10词(包括结束标点符号),我们正在过滤到翻译成”I am”或”He is”等形式的句子.(考虑到先前替换了撇号).
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# 加载文本文件切分成行,并切分成单词对:
# 文本归一化, 按照长度和内容过滤
# 从成对的句子中制作单词列表
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    # print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    # print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    # print("Counted words:")
    # print(input_lang.name, input_lang.n_words)
    # print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))
