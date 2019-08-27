# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:59:57 2019

@author: Zhenlin
"""

from n_gram_model import NGram
from preprocess_sohu_news import preprocess_sohu_news
import pickle
import config


news=preprocess_sohu_news(config.sohu_path)

text=[]
for one_news in news:
    chars=[]
    for c in one_news:
        if len(c.strip())>0:
            chars.append(c.strip())
    text.append(chars)

ng = NGram(text, config.num_gram)


f = open(config.model_path, 'wb')
pickle.dump(ng, f)
f.close()
