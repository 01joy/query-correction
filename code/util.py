#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:24:49 2019

@author: bytedance
"""

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False
    
def clean_data(texts, dest_path):
    n=len(texts)
    fout=open(dest_path,'w')
    for k, line in enumerate(texts):
        print('%d/%d\n'%(k,n))
        t=[]
        for c in line:
            if c.strip()!='':
                t.append(c)
        if len(t) > 0:
            fout.write(''.join(t)+'\n')
    fout.close()