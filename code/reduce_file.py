import config

fin=open(config.sohu_big_path,encoding='utf-8',errors='ignore')
fout=open(config.sohu_path,'w')

for i,line in enumerate(fin):
	fout.write(line)
	if i>=config.num_corpus and config.end_flag in line:
		break
fin.close()
fout.close()

