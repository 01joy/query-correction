# -*- coding: UTF-8 -*-
import json
import readline


def find_words(ch, w2p_path, p2w_path, pp_err_path):
    
    with open(w2p_path, "r", encoding = 'utf-8') as f:
        word2pinyin = json.load(f)
    with open(p2w_path, 'r', encoding = 'utf-8')  as f:
        pinyin2word = json.load(f)
    with open(pp_err_path, 'r', encoding = 'utf-8') as f:
        pp_js = json.load(f)
    if ch not in pp_js.keys():
        pp_js = {}
    else:
        pp_js = pp_js[ch]

#print(pp_js)
    
    if ch not in word2pinyin.keys():
        pinyins = []
        words = [ch]
    else:
        pinyins = word2pinyin[ch]
        words = []
    for pinyin in pinyins:
        words.extend(pinyin2word[pinyin])
    words = list(set(words))

    rnt = {}
    sm = 0.0;
    for word in words:
        rnt[word] = 1.0
        if word in pp_js.keys():
            rnt[word] = rnt[word] + pp_js[word]
        sm = sm + rnt[word]

    if(sm == 0):
        print("sm == 0")
        exit()

    for word in words:
        rnt[word] = rnt[word] / sm

    return rnt

if __name__ == "__main__":

    w2p = "../data/w2p_v2.json"
    p2w = '../data/p2w_v2.json'
    pp_err_path = '../data/pp_err.json'

    st = "你在干吗"

    for ch in st:
        words = find_words(ch, w2p, p2w, pp_err_path)
        print(words)
#print(words['他'])
