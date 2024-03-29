# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:45:08 2019

@author: Zhenlin
"""

import codecs

BLOCKSIZE = 504857600 # or some other, desired size in bytes

def convert_file_to_utf8(sourceFileName,targetFileName,sourceEncoding):

    with codecs.open(sourceFileName, "r", sourceEncoding, errors='ignore') as sourceFile:
        with codecs.open(targetFileName, "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)
                
                
if __name__ == '__main__':
    sourceFileName=r'/Users/bytedance/Downloads/news_sohusite_xml.dat'
    targetFileName=r'/Users/bytedance/Downloads/news_sohusite_xml-utf8.dat'
    sourceEncoding='gb2312'
    convert_file_to_utf8(sourceFileName,targetFileName,sourceEncoding)
