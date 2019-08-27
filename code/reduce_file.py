big=r'/root/data/news_sohusite_xml.dat'
small=r'../data/news_sohusite_xml.200000.dat'
flag='</doc>'
n=200000

fin=open(big,encoding='gb2312',errors='ignore')
fout=open(small,'w')

for i,line in enumerate(fin):
	fout.write(line)
	if i>=n and flag in line:
		break
fin.close()
fout.close()

