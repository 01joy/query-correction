# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:06:34 2019

@author: Zhenlin
"""

from preprocess_sohu_news import preprocess_sohu_news
# Preprocess the tokenized text for 3-grams language modelling
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE


class NGram:
    train_text = []
    n = 3
    model = ''
    def __init__(self, train_text, n):
        self.train_text = train_text
        self.n = n
        
        train_data, padded_sents = padded_everygram_pipeline(self.n, self.train_text)
        self.model = MLE(self.n) # Lets train a 3-grams maximum likelihood estimation model.
        self.model.fit(train_data, padded_sents)
        
    def cal_sentence_prob(self, sentence):
        chars = [i for i in sentence]
        ans = 1.0
        m = len(sentence)
        
        for i in range(m):
            l = max(0, i - self.n + 1)
            r = i
            pre = chars[l:r]
            ans = ans * self.model.score(chars[i], pre)
        return ans
        
    
    
if __name__ == '__main__':
    
    sohu_path=r'C:\D\Hobby\Deeplearning\ByteCamp\query-correction\data\news_sohusite_xml.smarty.dat'
    
    news=preprocess_sohu_news(sohu_path)
    
    text=[]
    
    for one_news in news:
        chars=[]
        for c in one_news:
            if len(c.strip())>0:
                chars.append(c.strip())
        text.append(chars)

    ng = NGram(text, 3)
    
    sentence = '中国人'
    prob = ng.cal_sentence_prob(sentence)
    print(prob)
