# preprocess.py
# Further preprocess the wikipedia data. This will be important for 
# classical search algorithms like TF-IDF and BM25.
# Python 3.9
# Windows/MacOS/Linux

import os
# from collections import Counter
import copy
# import gc
import json
# import math
import string
from bs4 import BeautifulSoup
import faiss
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import num2words
# import numpy as np
# from tqdm import tqdm
from transformers import AutoTokenizer


def lowercase(text: str) -> str:
	'''
	Lowercase all characters in the text.
	@param: text (str), the text that is going to be lowercased.
	@return: returns the text with all characters in lowercase.
	'''
	return text.lower()


def remove_punctuation(text: str) -> str:
	'''
	Replace all punctuation with whitespace (" ") or empty space ("")
		depending on the character.
	@param: text (str), the text that is going to have its punctuation
		removed/modified.
	@return: returns the text with all punctuation a characters 
		replaced with whitespace or emptyspace.
	'''
	empty_characters = ",'"
	for char in string.punctuation:
		if char in empty_characters:
			text = text.replace(char, "")
		else:
			text = text.replace(char, " ")

	return text


def remove_stopwords(text: str) -> str:
	'''
	Remove all stopwords from the text.
	@param: text (str), the text that is going to have its stop words
		removed.
	@return: returns the text without any stop words.
	'''
	stop_words = set(stopwords.words("english"))
	words = word_tokenize(text)
	# new_text = ""
	# for word in words:
	# 	if word not in stop_words and len(word) > 1:
	# 		new_text = new_text + " " + word
	# text = new_text
	text = " ".join(
		[
			word for word in words 
			if word not in stop_words and len(word) > 1
		]
	)

	return text


def convert_numbers(text: str) -> str:
	'''
	Convert all numbers in the text from their numerical representation
		to their written expanded representation (1 -> one).
	@param: text (str), the text that is going to have its numbers 
		converted.
	@return: returns the text with all of its numbers expanded to their
		written expanded representations.
	'''
	words = word_tokenize(text)
	# new_text = ""
	# for w in words:
	# 	try:
	# 		w = num2words(int(w))
	# 	except:
	# 		a = 0
	# 	new_text = new_text + " " + w
	# new_text = np.char.replace(new_text, "-", " ")
	# text = new_text
	text = " ".join(
		[
			num2words(int(word)) for word in words
			if word.isdigit()
		]
	)

	return text


def lemmatize(text: str) -> str:
	'''
	Lemmatize the words in the text.
	@param: text (str), the text that is going to be lemmatized.
	@return: returns the lemmatized text.
	'''
	lemmatizer = WordNetLemmatizer()
	words = word_tokenize(text)
	text = " ".join(
		[lemmatizer.lemmatize(word) for word in words]
	)

	return text


def stem(text) -> str:
	'''
	Stem the words in the text.
	@param: text (str), the text that is going to be stemmed.
	@return: returns the stemmed text.
	'''
	stemmer = PorterStemmer()
	words = word_tokenize(text)
	text = " ".join(
		[stemmer.stem(word) for word in words]
	)
	
	return text


def bow_preprocessing(text: str, return_word_freq: bool=False):
	'''
	Preprocess the text to yield a bag of words used in the text. Also
		compute the word frequencies for each word in the bag of words
		if specified in the arguments.
	@param: text (str), the raw text that is to be processed into a bag
		of words.
	@param: return_word_freq (bool), whether to return the frequency of
		each word in the input text.
	@return: returns a tuple (bag_of_words: List[str]) or 
		(bag_of_words: List[str], freq: dict[str: int]) depending on the 
		return_word_freq argument.
	'''
	# Perform the following text preprocessing in the following order:
	# 1) lowercase
	# 2) remove punctuation
	# 3) remove stop words
	# 4) convert numbers
	# 5) lemmatize
	# 6) stem
	# 7) remove punctuation
	# 8) convert numbers
	# 9) stem
	# 10) remove punctuation
	# 11) remove stopwords
	# Note how some of these steps are repeated. This is because 
	# previous steps may have introduced conditions that were 
	# previously handled. However, this order of operations is both
	# optimal and firm.
	text = lowercase(text)
	text = remove_punctuation(text)
	text = remove_stopwords(text)
	text = convert_numbers(text)
	text = lemmatize(text)
	text = stem(text)
	text = remove_punctuation(text)
	text = convert_numbers(text)
	text = stem(text)
	text = remove_punctuation(text)
	text = remove_stopwords(text)

	# NOTE:
	# Number conversion, lemmatizaton, and stemming do not seem to be
	# hard/firm requirements to perform the bag of words preprocessing.
	# They do offer some improvement for the search component
	# downstream so they are left in.
	# One aspect that does have me wary of performing this is that
	# these steps rely on third party libraries to do their respective
	# transformations. Since I want to convert this project to JS, I am
	# trying to minimize the number of third party packages since I
	# have to find a JS counterpart that works as close as possible.

	# Isolate the set of unique words in the remaining processed text.
	bag_of_words = list(set(word_tokenize(text)))

	# Return just the bag of words if return_word_freq is False.
	if not return_word_freq:
		return tuple([bag_of_words])

 	# Record each word's frequency in the processed text.
	word_freqs = dict()
	words = word_tokenize(text)
	for word in bag_of_words:
		word_freqs[word] = words.count(word)

	# Return the bag of words and the word frequencies.
	return tuple([bag_of_words, word_freqs])


def vector_preprocessing(article_text: str, context_length: int, tokenizer: AutoTokenizer):
	'''
	Preprocess the text to yield a list of chunks of the text. Each 
		chunk is the longest possible set of text that can be passed to
		the embedding model tokenizer.
	@param: text (str), the raw text that is to be processed into a bag
		of words.
	@param: context_length (int), the maximum number of tokens that
	 	will be accepted by the embedding model.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@return: returns a List[str] of the text sliced into chunks that
		are acceptable size to the embedding model.
	'''
	# Split the text but newline characters ("\n").
	line_splits = article_text.split("\n")
	num_splits = len(line_splits)

	# Subtract the number of splits from the context length so that we
	# do not forget about the newline characters removed in the split
	# during the tokenization.
	true_context_length = context_length - num_splits


	return 


def main() -> None:
	'''
	Main method. Process the individual wikipedia articles from their
		xml files to create document to word and word to document 
		mappings for faster bag of words processing during classical
		search (TF-IDF/BM25) and vector databases for vector search.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	###################################################################
	# VERIFY DATA FILES
	###################################################################
	# Check for WikipediaEnDownload submodule and additional necessary
	# folders to be initialized.
	submodule_dir = "./WikipediaEnDownload"
	submodule_data_dir = os.path.join(submodule_dir, "WikipediaData")
	if not os.path.exists(submodule_dir):
		print(f"WikipediaEnDownload submodule not initialized.")
		print(f"Please initialized submodule with 'git submodule update --init --recursive'")
		exit(1)
	elif not os.path.exists(submodule_data_dir):
		print(f"WikipediaEnDownload submodule has not extracted any articles from the downloader.")
		print(f"Follow the README.md in the WikipediaEnDownload submodule for instructions on how to download and extract articles from wikipedia.")
		exit(1)
	
	# NOTE:
	# I tried to make this cleaner but python would throw an error on
	# on the os.listdir() line for the submodule data directory if that
	# directory did not exist. Therefore, it made it impossible to
	# define data_files before checking for the existance of the 
	# required submodule data directory.
	data_files = sorted(
		[
			os.path.join(submodule_data_dir, file) 
			for file in os.listdir(submodule_data_dir)
			if file.endswith(".xml")
		]
	)
	if len(data_files) == 0:
		print(f"WikipediaEnDownload submodule has not extracted any articles from the downloader.")
		print(f"Follow the README.md in the WikipediaEnDownload submodule for instructions on how to download and extract articles from wikipedia.")
		exit(1)

	###################################################################
	# NLTK SETUP
	###################################################################
	# Download packages from nltk.
	# nltk.download("stopwords")

	###################################################################
	# EMBEDDING MODEL SETUP
	###################################################################
	# Check for embedding module files WikipediaEnDownload submodule and additional necessary
	# folders to be initialized.
	# 

	###################################################################
	# FILE PREPROCESSING
	###################################################################
	# Initialize a dictionary to keep track of the word to documents
	# and documents to words mappings.
	word_to_doc = dict()
	doc_to_word = dict()

	# Iterate through each file and preprocess it.
	for idx, file in enumerate(data_files):
		# Read in the file.
		with open(file, "r") as f:
			raw_text = f.read()

		# Load the raw text into a beautifulsoup object and extract the
		# <title> and <text> tags.
		soup = BeautifulSoup(raw_text, "lxml")
		title_tag = soup.find("title")
		text_tag = soup.find("text")

		# Combine the title and text tag texts together.
		article_text = title_tag.get_text() + "\n\n" + text_tag.get_text()
		article_text_bow = copy.deepcopy(article_text)
		article_text_v_db = copy.deepcopy(article_text)

		###############################################################
		# BAG OF WORDS
		###############################################################
		# Create a bag of words for each article (xml) file.
		xml_bow, xml_word_freq = bow_preprocessing(article_text_bow)

		# Update word to document map.
		for word in xml_bow:
			if word in list(word_to_doc.keys()):
				word_to_doc[word].append(file)
			else:
				word_to_doc[word] = [file]

		# Update the document to words map.
		doc_to_word[file] = xml_word_freq

		###############################################################
		# VECTOR EMBEDDINGS
		###############################################################
		# Pass the article 
		xml_chunks = vector_preprocessing(article_text_v_db)

		# Embed chunks and write them to vector storage.
		for chunk in xml_chunks:
			pass


		# Perform garbage collection.
		# gc.collect()

		# NOTE:
		# Checkpoint interval is set to 50,000 (articles/documents)
		# since the number of articles expected in the english
		# wikipedia dataset is expected to be in the order of
		# millions. Going by article/document makes the data sharding
		# and retrieval much more straightforward.
		# I may change the interval if this creates an unreasonable
		# number of files. Alternatively, NodeJS is not able to handle
		# very large files. It has a hard limit on opening files over
		# 1GB on 64-bit systems or 512MB on 32-bit systems (error on
		# the side of 64-bit in this day and age).

		# Checkpoint save current status.
		if idx > 0 and idx % 50_000 == 0:
			pass

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()