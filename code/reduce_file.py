big=r'/root/data/news_sohusite_xml.dat'
small=r'/root/data/news_sohusite_xml.100000.dat'
flag='</doc>'
n=100000

fin=open(big)
fout=open(small,'w')

for i,line in enumerate(fin):
	fout.write(line)
	if i>=n and flag in line:
		break
fin.close()
fout.close()

