import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from process_html import *
import os
import re

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

train = pd.read_csv('../data/train.csv')

sponsored = train[train.sponsored == 1]
organic = train[train.sponsored == 0]

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def feature_headers():
	return ['num_images', 'num_p', 'num_links', 'sponsored']

def urlid(filename):
	return filename.split('_')[0]

# replace multiple spaces with just one space
def clean_text(text):
	print text
	return re.sub(r'\s+', ' ', text)

tfidf = TfidfVectorizer(stop_words='english')
def fit_idf_from_dir(dir_name, batch_size=10):
	filenames = os.listdir(dir_name)
	num_files = len(filenames[:20])
	corpus = []

	for i in range(0, num_files, batch_size):
		batch_filenames = filenames[i:min(i+batch_size, num_files)]

		texts = []
		for j in range(len(batch_filenames)):
			filename = batch_filenames[j]
			if filename == '.DS_Store':
				continue
			text = ' '.join(parse_text_only(dir_name + filename, urlid(filename)))
			text = clean_text(text)
			texts.append(text)
		print texts


		# full_text = ' '.join(doc['text'])
		# fit_idf(full_text)

def fit_idf(text):
	tfidf.fit()

# only perform transform
def tfidf_for_text(text):
	terms = tfidf.fit_transform(tokenize(text))
	feature_names = tfidf.get_feature_names()
	print tfidf.idf_
	# for i in terms.nonzero()[1]:
	# 	print feature_names[i], ' - ', terms[0, i]

def features_for_directory(dir_name):
	f = open('../data/features.csv', 'w')	# erase previous entries
	f.write(','.join(feature_headers()) + '\n')
	
	filenames = os.listdir(dir_name)
	for filename in filenames[:2]:	
		sponsored = train[train.file == filename].sponsored.values
		# file is not in training set
		if sponsored.size == 0:	
			continue

		sponsored = sponsored[0]
		doc = parse_page(dir_name + filename, urlid(filename))
		tfidf_for_text(' '.join(doc['text']))
		
		num_images, num_p, num_links = len(doc['images']), len(doc['text']), len(doc['links'])
		feature_row = ','.join(map(lambda k: str(k), [num_images, num_p, num_links, sponsored]))
		f.write(feature_row + '\n')
	f.close()

if __name__ == '__main__':
	# features_for_directory('../data/2/')
	fit_idf_from_dir('../data/2/')

