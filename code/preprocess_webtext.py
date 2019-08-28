# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:37:08 2019

@author: Zhenlin
"""
import util
import config
import json

start_mark='"text":'
def preprocess_webtext(text_path):
        
    fin=open(text_path,encoding='utf-8',errors='ignore')
    lines=fin.readlines()
    fin.close()
    
    answers=[]
    for line in lines:
        linestr=line.strip()
        contents=json.loads(linestr)
        
        if "content" in contents:
            ans= contents["content"].replace('\\n','').replace('\\r','')
            ans=''.join(ans.splitlines())
            if len(ans)>1:
                answers.append(ans)
    return answers

if __name__ == '__main__':
    answers=preprocess_webtext(config.webtext_path)
    util.clean_data(answers,config.webtext_clean_path,' ')
    
