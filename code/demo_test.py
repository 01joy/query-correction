# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:59:57 2019

@author: Zhenlin
"""

#from n_gram_model import NGram
#from preprocess_sohu_news import preprocess_sohu_news
from trans import find_words, find_words_quanpin
import copy
import pickle
import readline
import config
import util

import kenlm
model = kenlm.Model(config.model_path)

correct_cnt = 0

while True:
    sentence = input('please input your query, q for quit:')
    print(sentence)
    if sentence.strip() == 'q':
        break
    sentence = [i for i in sentence]
    corrections=[]
    
    l,r=util.has_letters(sentence)
    if l>-1 and r>-1:
        sent = sentence[:l]+['']+sentence[r:]
        cands=find_words_quanpin(''.join(sentence[l:r]))
        for cand, char_prob in cands.items():
            sent[l] = cand
            corr = ' '.join(sent)
            lm_prob = model.score(corr, bos = True, eos = True)
            lm_prob = pow(10, lm_prob)
#            print(corr,pinyin_prob,lm_prob)
            corrections.append([corr, char_prob * lm_prob])
    else:
        for i,c in enumerate(sentence):
            sent=copy.deepcopy(sentence)
            cands = find_words(c)
            for cand, pinyin_prob in cands.items():
                sent[i]=cand
                corr = ' '.join(sent)
                lm_prob = model.score(corr, bos = True, eos = True)
                lm_prob = pow(10, lm_prob)
    #            print(corr,pinyin_prob,lm_prob)
                corrections.append([corr, pinyin_prob * lm_prob])
    
    corrections=sorted(corrections,key=lambda x:x[1],reverse=True)

    result = ''.join(corrections[0][0].split())
    origin = ''.join(sentence)

    if result == origin:
        correct_cnt = correct_cnt + 1
    print(result)
    print(result)
    print(correct_cnt)
#for i in range(min(len(corrections), config.num_show)):
#print('%d\t%s\t%e'%(i, corrections[i][0], corrections[i][1]))
    
    
