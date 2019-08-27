#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:01:26 2019

@author: bytedance
"""

src=r'/Users/bytedance/Downloads/nlpcc2018+hsk/valid.src'
trg=r'/Users/bytedance/Downloads/nlpcc2018+hsk/valid.trg'

src2=r'/Users/bytedance/Downloads/nlpcc2018+hsk/valid2.src'
trg2=r'/Users/bytedance/Downloads/nlpcc2018+hsk/valid2.trg'


pyep=r'/Users/bytedance/Downloads/nlpcc2018+hsk/pinyin_error_prob2.csv'

from pypinyin import lazy_pinyin

#lazy_pinyin('中心')

fs=open(src)
slines=fs.readlines()
fs.close()

ft=open(trg)
tlines=ft.readlines()
ft.close()

fs2=open(src2,'w')
ft2=open(trg2,'w')

mpt2f={}

for s,t in zip(slines,tlines):
    n1=len(s)
    n2=len(t)
    if n1!=n2:
        continue
    error=0
    errorid=-1
    i=-1
    for u,v in zip(s,t):
        i+=1
        if u!=v:
            error+=1
            errorid=i
    if error == 1:
        sc=s[errorid]
        tc=t[errorid]
        p1=lazy_pinyin(sc)
        p2=lazy_pinyin(tc)
        if p1==p2:
            fs2.write(s.replace(' ',''))
            ft2.write(t.replace(' ',''))
            if tc not in mpt2f:
                mpt2f[tc]={}
            if sc not in mpt2f[tc]:
                mpt2f[tc][sc]=0
            mpt2f[tc][sc]+=1

fs2.close()
ft2.close()

fout=open(pyep,'w')
fout.write('true,false,num\n')
for t,m in mpt2f.items():
    for f,n in m.items():
        fout.write('%s,%s,%d\n'%(t,f,n))
fout.close()