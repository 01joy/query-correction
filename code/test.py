# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:59:57 2019

@author: Zhenlin
"""

#from n_gram_model import NGram
#from preprocess_sohu_news import preprocess_sohu_news
from trans import find_words
import copy
import pickle
import readline

N = 3 # n-gram
model_path=r'%d-gram.model'%N
w2p = "../data/w2p_v2.json"
p2w = '../data/p2w_v2.json'
pp_err_path = '../data/pp_err.json'

M = 5 # print list number

f = open(model_path, 'rb')
classifier = pickle.load(f)
f.close()

while True:
    sentence = input('please input your query, q for quit:')
    print(sentence)
    if sentence.strip() == 'q':
        break
    sentence = [i for i in sentence]
    corrections=[]
    
    for i,c in enumerate(sentence):
        sent=copy.deepcopy(sentence)
        cands = find_words(c, w2p, p2w, pp_err_path)
        for cand, pinyin_prob in cands.items():
            sent[i]=cand
            corr = ''.join(sent)
            corrections.append([corr,pinyin_prob*classifier.cal_sentence_prob(corr)])
    
    corrections=sorted(corrections,key=lambda x:x[1],reverse=True)
    for i in range(min(len(corrections), M)):
        print('%d\t%s\t%e'%(i, corrections[i][0], corrections[i][1]))
    
    
