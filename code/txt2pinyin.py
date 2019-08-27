#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:54:08 2019

@author: bytedance
"""
from preprocess_sohu_news import preprocess_sohu_news
from pypinyin import lazy_pinyin

sohu_path=r'/Users/bytedance/Downloads/news_sohusite_xml-utf8-10000.dat'

txt=r'/Users/bytedance/Downloads/news_sohusite_xml-utf8.txt'
pinyin=r'/Users/bytedance/Downloads/news_sohusite_xml-utf8.pinyin'

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False
    
news=preprocess_sohu_news(sohu_path)

ftxt=open(txt,'w')
fpy=open(pinyin,'w')

for k, line in enumerate(news):
    print('%d/%d\n'%(k,len(news)))
    t=[]
    p=[]
    for c in line:
        if is_chinese(c):
            t.append(c)
            p.append(lazy_pinyin(c))
    if len(t) == len(p):
        ftxt.write(' '.join(t)+'\n')
        fpy.write(' '.join([i[0] for i in p])+'\n')
ftxt.close()
fpy.close()