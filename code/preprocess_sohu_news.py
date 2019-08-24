# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:37:08 2019

@author: Zhenlin
"""

def preprocess_sohu_news(sohu_path):
    fin=open(sohu_path,encoding='gb2312',errors='ignore')
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
    sohu_path=r'C:\D\Hobby\Deeplearning\ByteCamp\query-correction\data\news_sohusite_xml.smarty.dat'
    news=preprocess_sohu_news(sohu_path)