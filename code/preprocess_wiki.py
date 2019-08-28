# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:37:08 2019

@author: Zhenlin
"""
import util
import config

start_mark='"text":'
def preprocess_wiki(wiki_folder):
    allfiles=util.absoluteFilePaths(wiki_folder)
    
    wikis=[]
    for path in allfiles:
        
        fin=open(path,encoding='utf-8',errors='ignore')
        lines=fin.readlines()
        fin.close()
        
        for line in lines:
            linestr=line.strip()
            l=linestr.index(start_mark)+len(start_mark)+2
            txt=linestr[l:-4].strip()
            txt=txt.replace('\\n')
            txt=''.join(txt.splitlines())
            if len(txt)>1:
                wikis.append(txt)
    return wikis

if __name__ == '__main__':
    wikis=preprocess_wiki(config.wiki_folder)
    util.clean_data(wikis,config.wiki_file,' ')
    
