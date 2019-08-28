#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:13:54 2019

@author: bytedance
"""

num_gram = 3 # n-gram
num_show = 5 # print list number
num_corpus = 100000

root = r'/root/bytecamp2019'
sohu_path = r'%s/datasets/news_sohusite_xml-utf8-%d.dat'%(root, num_corpus)
model_path = r'%s/models/news_sohusite.binary'%(root)

p_err_path = '../data/pp_err.json'
w2p_path = "../data/w2p_v2.json"
p2w_path = '../data/p2w_v2.json'
pp_err_path = '../data/pp_err.json'
quanpin_path = '../data/quanpin.json'
same_stroke_path = '../data/same_stroke.json'
p_same_stroke = 0.5


sohu_big_path=r'%s/datasets/news_sohusite_xml-utf8.dat'%root
end_flag='</doc>'



wiki_folder = r'%s/datasets/wiki_zh'%root
wiki_file = r'%s/datasets/wiki_zh_clean_space.dat'%root

webtext_path = r'%s/datasets/webtext2019zh/web_text_zh_train.json'%root
webtext_clean_path = r'%s/datasets/web_text_zh_train_clean_space.dat'%root
