# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:37:08 2019

@author: Zhenlin
"""
import util
import config

def preprocess_sohu_news(sohu_path):
    fin=open(sohu_path,encoding='utf-8',errors='ignore')
    lines=fin.readlines()
    fin.close()
    
    mark1='<content>'
    mark2='</content>'
    
    news=[]
    for i in range(4,len(lines),6):
        content=lines[i].strip()[len(mark1):-len(mark2)]
        content=content.strip()
        if len(content)>1:
            news.append(content)
    return news

if __name__ == '__main__':
    sohu_path=r'/Users/bytedance/Documents/datasets/news_sohusite_xml-utf8-100000.dat'
    sohu_clean=r'/Users/bytedance/Documents/datasets/news_sohusite_clean-utf8-100000.dat'
    sohu_clean_space=r'/Users/bytedance/Documents/datasets/news_sohusite_clean_space-utf8-100000.dat'
    news=preprocess_sohu_news(sohu_path)
    util.clean_data(news,sohu_clean)
    util.clean_data(news,sohu_clean_space,' ')
    