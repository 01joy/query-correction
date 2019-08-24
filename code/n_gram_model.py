# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:06:34 2019

@author: Zhenlin
"""

from preprocess_sohu_news import preprocess_sohu_news

sohu_path=r'C:\D\Hobby\Deeplearning\ByteCamp\query-correction\data\news_sohusite_xml.smarty.dat'

news=preprocess_sohu_news(sohu_path)

text=[]

for one_news in news:
    chars=[]
    for c in one_news:
        if len(c.strip())>0:
            chars.append(c.strip())
    text.append(chars)

from nltk.util import ngrams
text_bigrams = [ngrams(sent, 2) for sent in text]
text_unigrams = [ngrams(sent, 1) for sent in text]