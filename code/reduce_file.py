n=10000
big=r'/Users/bytedance/Downloads/news_sohusite_xml-utf8.dat'
small=r'/Users/bytedance/Downloads/news_sohusite_xml-utf8-%d.dat'%n
flag='</doc>'

fin=open(big,encoding='utf-8',errors='ignore')
fout=open(small,'w')

for i,line in enumerate(fin):
	fout.write(line)
	if i>=n and flag in line:
		break
fin.close()
fout.close()

