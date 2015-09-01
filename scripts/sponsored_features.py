import pandas as pd
import numpy as np
from process_html import *
import os
import re

from bs4 import BeautifulSoup as bs

tag_types = ['p', 'img', 'meta', 'style', 'script']

f_features = open('../data/sponsored_tag_counts.csv', 'w')
# write feature column headers
headers = ['file']
for tag_type in tag_types:
	headers.append(tag_type + '_count')
	headers.append(tag_type + '_char_count')
f_features.write(','.join(headers) + '\n')

# loop through sponsored files to extract features
def extract_html_features(src_dir):
	f_sponsored = open('../data/sponsored.txt', 'r')
	for filename in f_sponsored.readlines():
		filename = filename.strip()
		soup = bs_from_file(os.path.join(src_dir, filename))
		features = [filename]

		for tag_type in tag_types:
			counts = parse_by_name(soup, tag_type)
			features.append(counts['count'])
			features.append(counts['char_count'])
		f_features.write(','.join(map(lambda k: str(k), features)) + '\n')

	f_features.close()

extract_html_features('../data/2')