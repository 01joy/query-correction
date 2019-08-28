# -*- coding: UTF-8 -*-

import re
import random
import copy
import json
import os

datasets_path = '../../../datasets/'
out_path = datasets_path + 'search_word_total_2.txt'

same_usually_path = '../../data/usually.txt'
same_stroke_path = '../../data/same_stroke.json'
same_pinyin_path = '../../data/quanpin.json'

with open(same_stroke_path, 'r', encoding = 'utf-8') as f:
    same_words = json.load(f)
with open(same_pinyin_path, 'r', encoding = 'utf-8') as f:
    pinyin_words = json.load(f)
with open(same_usually_path, 'r', encoding = 'utf-8') as f:
    usually_str = f.readline()
    usually_word = list(usually_str)

def del_not_chinese(origin):
    rnt = ""
    for ch in origin:
        if ch >= u'\u4e00' and ch <= u'\u9fa5':
            rnt = rnt + ch
    return rnt

def trans(origin):
    p = 0.35
    q = 0.35

    pos = random.randint(0, len(origin) - 1)
    dst = copy.deepcopy(origin)
    
    rnd = random.random()

    if dst[pos] in same_words.keys() and rnd < p:
        words = same_words[dst[pos]]
        word = random.choice(words)
        dst[pos] = word
    elif dst[pos] in pinyin_words.keys() and p <= rnd and rnd < p + q:
        words = pinyin_words[dst[pos]]
        word = random.choice(words)
        dst[pos] = word
    else:
        dst[pos] = usually_word[random.randint(0, len(usually_word) - 1)]
    
    return dst
    
def gen(origin_str):
    p = 0.5
    origin_str = del_not_chinese(origin_str)
    if len(origin_str) <= 1:
        return
    origin = list(origin_str)

    for k in range(2 * len(origin)):
        dst = trans(origin)
        if len(origin_str) >= 4 and random.random() < p:
            dst = trans(dst)
        
        if(len(origin_str) != len(''.join(dst))):
            continue

        with open(out_path, 'a', encoding = 'utf-8') as fo:
            print(origin_str + '\t' + ''.join(dst), file = fo)

M = 10000

def work_sogou(in_path, out_path):
    print('load' + in_path)
    with open(in_path, 'r', encoding = 'gb2312', errors = 'ignore') as f:
        lines = f.readlines()
    print('finished load' + in_path)
    sz = len(lines)
    cnt = 1
    for line in lines:
        origin_str = re.split('\[|\]', line)[1]
        gen(origin_str)
        cnt = cnt + 1
        if cnt % M == 0:
            print(str(cnt) + '/' + str(sz))

def work_wiki(in_path, out_path):
    print('load' + in_path)
    with open(in_path, 'r', encoding = 'gb2312', errors = 'ignore') as f:
        lines = f.readlines()
    print('finished load' + in_path)
    sz = len(lines)
    cnt = 1
    for line in lines:
        js = json.loads(line)
        origin_str = js['title']
        gen(origin_str)
        if cnt % M == 0:
            print(str(cnt) + '/' + str(sz))
            
def find_paths(directory):
    ans=[]
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            ans.append(os.path.abspath(os.path.join(dirpath, f)))
    return ans

if __name__ == '__main__':
    if os.path.exists(out_path):
        os.remove(out_path)
    open(out_path, 'w')

    in_path_sogou = datasets_path + 'pre_title/SogouQ/'
    in_path_wiki = datasets_path + 'pre_title/wiki_zh/'
    in_path_webtext = datasets_path + 'pre_title/webtext2019zh/'

    paths = find_paths(in_path_wiki)
    for lists in paths:
        path = os.path.join(in_path_wiki, lists)
#print(path)
        work_wiki(path, out_path)

    paths = find_paths(in_path_webtext)
    for lists in paths:
        path = os.path.join(in_path_webtext, lists)
#print(path)
        work_wiki(path, out_path)

    paths = find_paths(in_path_sogou)
    for lists in paths:
        path = os.path.join(in_path_sogou, lists)
#print(path)
        work_sogou(path, out_path)
