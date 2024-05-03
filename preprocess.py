# preprocess.py
# Further preprocess the wikipedia data. This will be important for 
# classical search algorithms like TF-IDF and BM25.
# Python 3.9
# Windows/MacOS/Linux

import os
import copy
import json
import math
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def main():
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
	data_files = [
		os.path.join(submodule_data_dir, file) 
		for file in os.listdir(submodule_data_dir)
		if file.endswith(".xml")
	]
	if len(data_files) == 0:
		print(f"WikipediaEnDownload submodule has not extracted any articles from the downloader.")
		print(f"Follow the README.md in the WikipediaEnDownload submodule for instructions on how to download and extract articles from wikipedia.")
		exit(1)

	###################################################################
	# NLTK SETUP
	###################################################################
	# Download packages from nltk.
	nltk.download("stopwords")
 
	# Load stop words.
	stop_words = set(stopwords.words("english"))

	# Iterate through each file and preprocess it.
	for file in data_files:
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
		xml_bow = set()

		# Lowercase all words in the text.
		article_text_bow = article_text_bow.lower()

		# Replace all punctuation with " " (whitespace) or "" empty
		# space depending on the character.
		empty_characters = ",'"
		for char in string.punctuation:
			if char in empty_characters:
				article_text_bow = article_text_bow.replace(char, "")
			else:
				article_text_bow = article_text_bow.replace(char, " ")

		# Remove stop words.
		words = word_tokenize(article_text_bow)
		new_text = ""
		for word in words:
			if word not in stop_words and len(word) > 1:
				new_text = new_text + " " + word
		article_text_bow = new_text

		# NOTE:
		# In the following example
		# https://github.com/dmmagdal/NeuralSearch/blob/main/
		# tf-idf_from_scratch/tf-idf_from_scratch.py
		# the num2words package https://pypi.org/project/num2words/ is
		# used to convert all numbers to their string counterparts
		# (1,001 -> one thousand and one). However, this step does not
		# seem to be applied for all examples in the repo
		# https://github.com/dmmagdal/NeuralSearch/ especially if you
		# look at the BM25 and TF-IDF implementations. I am not sure
		# how much of an improvement that including it will be and for
		# every extra python package I use in this implementation, I
		# will need to find its counterpart in JS for that
		# implementation.

		# Lemmatize then stem the words.


		# For each word, get the frequency of the word.
		xml_bow = set(article_text_bow.split(" "))


		###############################################################
		# VECTOR EMBEDDINGS
		###############################################################
		line_splits = article_text.split("\n")


	# Exit the program.
	exit(0)


def bow_preprocessing(article_text):
	return


def vector_preprocessing(article_text):
	return


if __name__ == '__main__':
	main()