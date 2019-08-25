# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:59:57 2019

@author: Zhenlin
"""

from n_gram_model import NGram
from preprocess_sohu_news import preprocess_sohu_news
from trans import find_words
import copy

sohu_path=r'C:\D\Hobby\Deeplearning\ByteCamp\query-correction\data\news_sohusite_xml.smarty.dat'
w2p = "../data/word2pinyin.json"
p2w = '../data/pinyin2word.json'
sentence = '中锅人'


news=preprocess_sohu_news(sohu_path)

text=[]
for one_news in news:
    chars=[]
    for c in one_news:
        if len(c.strip())>0:
            chars.append(c.strip())
    text.append(chars)

ng = NGram(text, 3)

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
print(corrections[0])