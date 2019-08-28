from multiprocessing.dummy import Pool as ThreadPool

import re
import random
import copy
import json
import os

import time

datasets_path = '../../data/'
#datasets_path = '../../../datasets/'
out_path = datasets_path + 'title.txt'

def del_not_chinese(origin):
    rnt = ""
    for ch in origin:
        if ch >= u'\u4e00' and ch <= u'\u9fa5':
            rnt = rnt + ch
    return rnt

def gen_sogou(line):
    return del_not_chinese(re.split('\[|\]', line)[1])

def gen_wiki(line):
    return del_not_chinese(json.loads(line)['title'])
    
def work_sogou(path):
    print('begin load' + path)
    with open(path, 'r', encoding = 'gb2312', errors = 'ignore') as f:
        lines = f.readlines()
    print('finished load' + path)
    
    pool = ThreadPool()
    results = pool.map(gen_sogou, lines)
    pool.close()
    pool.join()
    
    print('begin output' + path)
    with open(out_path, 'w', encoding = 'utf-8') as f:
        for st in results:
            if len(st) == 0:
                continue
            f.write(st)
            f.write('\n')
    print('finished output' + path)
    
def work_wiki(path):
    print('begin load' + path)
    with open(path, 'r', encoding = 'gb2312', errors = 'ignore') as f:
        lines = f.readlines()
    print('finished load' + path)
    
    pool = ThreadPool()
    results = pool.map(gen_wiki, lines)
    pool.close()
    pool.join()
    
    print('begin output' + path)
    with open(out_path, 'w', encoding = 'utf-8') as f:
        for st in results:
            f.write(st)
            f.write('\n')
    print('finished output' + path)
    
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
        work_wiki(path)

    paths = find_paths(in_path_webtext)
    for lists in paths:
        path = os.path.join(in_path_webtext, lists)
#print(path)
        work_wiki(path)

    paths = find_paths(in_path_sogou)
    for lists in paths:
        path = os.path.join(in_path_sogou, lists)
#print(path)
        work_sogou(path)
