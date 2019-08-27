# -*- coding: UTF-8 -*-
import json
import readline
import config


def find_words(ch):
    with open(config.w2p_path, "r", encoding = 'utf-8') as f:
        word2pinyin = json.load(f)
    with open(config.p2w_path, 'r', encoding = 'utf-8')  as f:
        pinyin2word = json.load(f)
    with open(config.pp_err_path, 'r', encoding = 'utf-8') as f:
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
        rnt[word] = rnt[word] / sm * (1 - config.p_same_stroke)

    with open(config.same_stroke_path, 'r', encoding = 'utf-8') as f:
        same_js = json.load(f)
    if ch in same_js:
        same_lst = same_js[ch]
        for c in same_lst:
            if c in rnt:
                rnt[c] = rnt[c] + 1.0 / len(same_lst) * config.p_same_stroke
            else:
                rnt[c] = 1.0 / len(same_lst) * config.p_same_stroke
#print(same_lst)
        

    return rnt

def find_words_quanpin(st):
    with open(config.quanpin_path, 'r', encoding = 'utf-8') as f:
        qp = json.load(f)
    if st not in qp.keys():
        return { st : 1.0 }
    else:
        return qp[st]

if __name__ == "__main__":

    '''
    st = "zai"
    print(find_words_quanpin(st))
    st = "ni"
    print(find_words_quanpin(st))
    st = "zhai"
    print(find_words_quanpin(st))

    st = "他"
    for ch in st:
        words = find_words(ch)
        print(words)
    '''
    ch = '大'
    print(find_words(ch))

#print(words['他'])
