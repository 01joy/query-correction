import json


def find_words(ch, w2p, p2w):
    
    with open(w2p, "r", encoding = 'utf-8') as f:
        word2pinyin = json.load(f)
    with open(p2w, 'r', encoding = 'utf-8') as f:
        pinyin2word = json.load(f)
    
    
    if ch not in word2pinyin.keys():
        pinyins = []
        words = [ch]
    else:
        pinyins = word2pinyin[ch]
        words = []
    for pinyin in pinyins:
        words.extend(pinyin2word[pinyin])
    words = {}.fromkeys(words).keys()
    return words

if __name__ == "__main__":

    w2p = "../data/w2p_v2.json"
    p2w = '../data/p2w_v2.json'
    str = "孙点生"
    for ch in str:
        words = find_words(ch, w2p, p2w)
        print(words)
