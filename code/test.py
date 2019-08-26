# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:59:57 2019

@author: Zhenlin
"""

from n_gram_model import NGram
from preprocess_sohu_news import preprocess_sohu_news
from trans import find_words
import copy

sohu_path=r'../data/news_sohusite_xml.100000.dat'
w2p = "../data/word2pinyin.json"
p2w = '../data/pinyin2word.json'
N = 3 # n-gram
M = 5 # print list number
#sentence = '刘得华'

news=preprocess_sohu_news(sohu_path)

text=[]
for one_news in news:
    chars=[]
    for c in one_news:
        if len(c.strip())>0:
            chars.append(c.strip())
    text.append(chars)

ng = NGram(text, N)

while True:
    sentence = input('please input your query:')
    sentence = [i for i in sentence]
    corrections=[]
    
    for i,c in enumerate(sentence):
        sent=copy.deepcopy(sentence)
        cands = find_words(c, w2p, p2w)
        for cand in cands:
            sent[i]=cand
            corr = ''.join(sent)
            corrections.append([corr,ng.cal_sentence_prob(corr)])
    
    corrections=sorted(corrections,key=lambda x:x[1],reverse=True)
    for i in range(min(len(corrections), M)):
        print('%d\t%s\t%e'%(i, corrections[i][0], corrections[i][1]))

    
