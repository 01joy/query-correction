# -*- coding: UTF-8 -*-
from multiprocessing.dummy import Pool as ThreadPool

import re
import random
import copy
import json
import os

#datasets_path = '../../../datasets/'
datasets_path = '../../data/'
out_path = datasets_path + 'search_word.txt'

same_usually_path = '../../data/usually.txt'
same_stroke_path = '../../data/same_stroke.json'
w2p_path = '../../data/w2p_v2.json'
p2w_path = '../../data/p2w_v2.json'

with open(same_stroke_path, 'r', encoding = 'utf-8') as f:
    same_words = json.load(f)
with open(w2p_path, 'r', encoding = 'utf-8') as f:
    w2p = json.load(f)
with open(p2w_path, 'r', encoding = 'utf-8') as f:
    p2w = json.load(f)
with open(same_usually_path, 'r', encoding = 'utf-8') as f:
    usually_str = f.readline()
    usually_word = list(usually_str)
    
def trans(origin):
    p = 0.35
    q = 0.35
    
    pos = random.randint(0, len(origin) - 1)
    dst = copy.deepcopy(origin)
    
    #print(dst)
    
    rnd = random.random()
    type = str(0)
    
    if dst[pos] in same_words.keys() and rnd < p:
        words = same_words[dst[pos]]
        word = random.choice(words)
        dst[pos] = word
        type = str(1)
    elif dst[pos] in w2p.keys() and p <= rnd and rnd < p + q:
        pinyin = w2p[dst[pos]][0]
        word = random.choice(p2w[pinyin])
        dst[pos] = word
        type = str(2)
    else:
        dst[pos] = usually_word[random.randint(0, len(usually_word) - 1)]
        type = str(3)
        
    #print(dst)
    return ''.join(dst), type
    
def gen(origin_str):
    origin_str = origin_str.strip()
    
    p = 0.5
    if len(origin_str) <= 1:
        return []
    origin = list(origin_str)

    rnt = []
    for k in range(2 * len(origin)):
        mid = [origin_str]
        dst, type = trans(origin)
        mid.append(dst)
        mid.append(type)
        if len(origin_str) >= 4 and random.random() < p:
            dst, tpye = trans(list(mid[1]))
            mid[1] = dst
            mid.append(type)
        
        if(len(origin_str) != len(mid[1])):
            continue
        
        #print(mid)
        rnt.append(mid)
    return rnt
            
if __name__ == '__main__':
    in_path = datasets_path + 'title.txt'
    print('begin load file.')
    with open(in_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    print('finished load file.')
    
    
    pool = ThreadPool()
    results = pool.map(gen, lines)
    pool.close()
    pool.join()
    '''
    results = []
    for line in lines:
        #print(line)
        results.append(gen(line))
    '''
    
    if os.path.exists(out_path):
        os.remove(out_path)
    open(out_path, 'w')
    
    
    with open(out_path, 'w', encoding = 'utf-8') as f:
        for wordss in results:
            for words in wordss:
                for word in words:
                    #print(word)
                    f.write(word)
                    f.write('\t')
                f.write('\n')
