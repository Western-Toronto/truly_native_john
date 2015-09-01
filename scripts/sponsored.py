import os
import pandas as pd
import numpy as np
import re

train = pd.read_csv('../data/train.csv')

def output_sponsored(dir_name):
	f = open('../data/sponsored.txt', 'w')
	filenames = os.listdir(dir_name)
	for filename in filenames:
		df = train[train.file == filename]
		if df.empty:
			continue
		if int(df.sponsored.iloc[0]) == 1:
			f.write(filename + '\n')

def reformat_sponsored(dir_name):
	lines = open('../data/sponsored.txt').readlines()
	for line in lines:
		filename = line.rstrip()
		html_name = re.sub('_html', '.html', filename.split('.txt')[0])
		os.system('cp ' + dir_name + filename + ' ../render/' + html_name)

if __name__ == '__main__':
	# output_sponsored('../data/2/')
	reformat_sponsored('../data/2/')
