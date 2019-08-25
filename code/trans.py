import json


def find_words(ch):
	if ch not in word2pinyin.keys():
		pinyins = []
		words = [ch]
	else:
		pinyins = word2pinyin[ch]
		words = []
	for pinyin in pinyins:
		words.extend(pinyin2word[pinyin])
	return words

if __name__ == "__main__":
	with open("../dat/word2pinyin.json", "r") as f:
		word2pinyin = json.load(f)
	with open('../dat/pinyin2word.json', 'r') as f:
		pinyin2word = json.load(f)

	str = "孙点生"
	for ch in str:
		words = find_words(ch)
		print(words)
