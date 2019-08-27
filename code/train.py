# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:59:57 2019

@author: Zhenlin
"""

from n_gram_model import NGram
from preprocess_sohu_news import preprocess_sohu_news
import pickle

N = 3 # n-gram
sohu_path=r'/Users/bytedance/Downloads/news_sohusite_xml-utf8-10000.dat'
model_path=r'%d-gram.model'%N

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


f = open(model_path, 'wb')
pickle.dump(ng, f)
f.close()
