import json

def find_word_pinyin(word_path, pinyin_path, out_path):
    rnt = {}
    with open(word_path, 'r', encoding = 'utf-8') as fw, open(pinyin_path, 'r', encoding = 'utf-8') as fp:
        for sw, sp in zip(fw, fp):
            lw = sw.split()
            lp = sp.split()
            for ww, wp in zip(lw, lp):
                if wp not in rnt.keys():
                    rnt[wp] = {}
                if ww not in rnt[wp].keys():
                    rnt[wp][ww] = 0.0
                rnt[wp][ww] = rnt[wp][ww] + 1.0
    for pinyin in rnt.keys():
        sm = 0.0
        for word in rnt[pinyin].keys():
            sm = sm + rnt[pinyin][word]
        for word in rnt[pinyin].keys():
            rnt[pinyin][word] = rnt[pinyin][word] / sm

    with open(out_path, 'w', encoding = 'utf-8') as f:
        json.dump(rnt, f)

if __name__ == '__main__':
#quanpin_word_path = '../data/quanpin_word.txt'
    quanpin_word_path = '../data/news_sohusite_xml-utf8.txt'
#quanpin_pinyin_path = '../data/quanpin_pinyin.txt'
    quanpin_pinyin_path = '../data/news_sohusite_xml-utf8.pinyin'
    out_path = '../data/quanpin.json'
    find_word_pinyin(quanpin_word_path, quanpin_pinyin_path, out_path)

