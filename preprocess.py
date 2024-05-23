# preprocess.py
# Further preprocess the wikipedia data. This will be important for 
# classical search algorithms like TF-IDF and BM25.
# Python 3.9
# Windows/MacOS/Linux

import argparse
from argparse import Namespace
import copy
import json
import math
import multiprocessing as mp
import os
import shutil
import string
from typing import List, Dict, Tuple

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import faiss
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from num2words import num2words
# import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def process_page(page: Tag | NavigableString) -> str:
	'''
	Format and merge the texts from the <title> and <text> tags in the
		current <page> tag.
	@param: page (Tag | NavigatableString), the <page> element that is
		going to be parsed.
	@return: returns the text from the <title> and <text> tags merged
		together.
	'''

	# Assert correct typing for the page.
	assert isinstance(page, Tag) or isinstance(page, NavigableString) or page is None,\
		"Expected page to be a Tag or NavigatableString."

	# Return empty string if page is none.
	if page is None:
		return ""

	# Locate the title and text tags (expect to have 1 of each per 
	# article/page).
	title_tag = page.find("title")
	text_tag = page.find("text")

	# Combine the title and text tag texts together.
	article_text = title_tag.get_text() + "\n\n" + text_tag.get_text()
	
	# Return the text.
	return article_text


def lowercase(text: str) -> str:
	'''
	Lowercase all characters in the text.
	@param: text (str), the text that is going to be lowercased.
	@return: returns the text with all characters in lowercase.
	'''
	return text.lower()


def replace_superscripts(text: str) -> str:
	'''
	Replace all superscripts depending on the character.
	@param: text (str), the text that is going to have its text 
		removed/modified.
	@return: returns the text with all superscript characters 
		replaced with regular numbers.
	'''
	superscript_map = {
		'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', 
		'⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
	}
	
	# result = ""
	# for char in text:
	# 	if char in superscript_map:
	# 		# result += '^' + superscript_map[char]
	# 		result += superscript_map[char]
	# 	else:
	# 		result += char
	# return result
	result = []
	i = 0
	while i < len(text):
		if text[i] in superscript_map:
			# Start of a superscript sequence.
			sequence = []
			while i < len(text) and text[i] in superscript_map:
				sequence.append(superscript_map[text[i]])
				i += 1

			# Join the sequence and prepend "^".
			result.append('^' + ''.join(sequence) + " ")
		else:
			result.append(text[i])
			i += 1

	return ''.join(result)


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
	# text = " ".join(
	# 	[
	# 		num2words(int(word)) for word in words
	# 		if word.isdigit()
	# 	]
	# )
	text = ""
	for word in words:
		if word.isdigit():
			if len(word) < 307:
				word = num2words(int(word))
			else:
				# Handles the edge case where the numerical text is 
				# greater than 1 x 10^307. Break apart the number and
				# process each half before merging them together.
				half_length = len(word) // 2
				word = num2words(int(word[:half_length])) + " " + num2words(int(word[half_length:]))

		text = text + " " + word

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
	# 4) remove superscripts
	# 5) convert numbers
	# 6) lemmatize
	# 7) stem
	# 8) remove punctuation
	# 9) convert numbers
	# 10) stem
	# 11) remove punctuation
	# 12) remove stopwords
	# Note how some of these steps are repeated. This is because 
	# previous steps may have introduced conditions that were 
	# previously handled. However, this order of operations is both
	# optimal and firm.
	text = lowercase(text)
	text = remove_punctuation(text)
	text = remove_stopwords(text)
	text = replace_superscripts(text)
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


def vector_preprocessing(article_text: str, config: Dict, tokenizer: AutoTokenizer):
	'''
	Preprocess the text to yield a list of chunks of the text. Each 
		chunk is the longest possible set of text that can be passed to
		the embedding model tokenizer.
	@param: text (str), the raw text that is to be processed into a bag
		of words.
	@param: config (dict), the configuration parameters. These 
		parameters detail important parts of the vector preprocessing
		such as context length.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@return: returns a List[str] of the text sliced into chunks that
		are acceptable size to the embedding model.
	'''
	# Split the text but newline characters ("\n").
	line_splits = article_text.split("\n")
	num_splits = len(line_splits)

	model_name = config["vector-search_config"]["model"]
	model_config = config["models"][model_name]
	context_length = model_config["max_tokens"]
	overlap = config["preprocessing"["token_overlap"]]

	# Subtract the number of splits from the context length so that we
	# do not forget about the newline characters removed in the split
	# during the tokenization.
	# true_context_length = context_length - num_splits

	# Splitting scheme:
	# 1) Split into paragraphs (split by newline "\n" character).
	# 2) Tokenize each paragraph
	#	a) if token sequence is too shoort, pad.
	#	b) if token sequence is too long, split up the tokens into 
	#		chunks with some overlap.
	for line in line_splits:
		# Skip empty entries.
		if line == "":
			continue

		# Perform an initial toeknization.
		tokens = tokenizer


	return 


def load_model(config: Dict) -> Tuple[AutoTokenizer, AutoModel]:
	# Check for the local copy of the model. If the model doesn't have
	# a local copy (the path doesn't exist), download it.
	model_name = config["vector-search_config"]["model"]
	model_config = config["models"][model_name]
	model_path = model_config["storage_dir"]
	
	# Check for path and that path is a directory. Make it if either is
	# not true.
	if os.path.exists(model_path) or os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)

	# Check for path the be populated with files (weak check). Download
	# the tokenizer and model and clean up files once done.
	if len(os.listdir(model_path)) == 0:
		print(f"Model {model_name} needs to be downloaded.")

		# Check for internet connection (also checks to see that
		# huggingface is online as well). Exit if fails.
		response = requests.get("https://huggingface.co/")
		if response.status_code != 200:
			print(f"Request to huggingface.co returned unexpected status code: {response.status_code}")
			print(f"Unable to download {model_name} model.")
			exit(1)

		# Create cache path folders.
		cache_path = model_config["cache_dir"]
		os.makedirs(cache_path, exist_ok=True)
		os.makedirs(model_path, exist_ok=True)

		# Load tokenizer and model.
		model_id = model_config["model_id"]
		tokenizer = AutoTokenizer.from_pretrained(
			model_id, cache_dir=cache_path
		)
		model = AutoModel.from_pretrained(
			model_id, cache_dir=cache_path
		)

		# Save the tokenizer and model to the save path.
		tokenizer.save_pretrained(model_path)
		model.save_pretrained(model_path)

		# Delete the cache.
		shutil.rmtree(cache_path)
	
	# Load the tokenizer and model.
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModel.from_pretrained(model_path)

	# Return the tokenizer and model.
	return tokenizer, model


def merge_mappings(results: List[List[Dict]]):
	assert len(results) == 3, "Expected results argument to be a tuple of length 3."
	assert all([isinstance(result, dict) for result in results], "Expected results argument to contain all dictionary objects.")
	word_to_docs, doc_to_words, chunk_to_docs = results
	pass


def multiprocess_articles(args: Namespace, device: str, file: str, pages: List[str], num_proc: int=1):
	'''
	Preprocess the text (in a multiple processors.
	@param: args (Namespace), the arguments passed in from the 
		terminal.
	@param: device (str), the name of the CPU or GPU device that the
		embedding model will use.
	@param: file (str), the filepath of the current file being
		procesze = sys.getsizeof(doc_to_word)
		# w2d_size = sys.getsizeof(word_to_doc)

		# GB_SIZE = 1024 * 1024 * 1024
		# if d2w_size // GB_SIZE > 1:
		# 	print(f"Document to word map has reached over 1GB in size")
		# 	exit()
		# elif w2d_size // GB_SIZE > 1:
		# 	print(f"Word to document map has reached over 1GB in size")
		# 	exit()sed.
	@param: pages (List[str]), the raw xml text that is going to be
		processed.
	@param: num_proc (int), the number of processes to use. Default is 
		1.
	@return: returns the set of dictionaries containing the necessary 
		data and metadata to index the articles.
	'''
	# Break down the list of pages into chunks.
	chunk_size = math.ceil(len(pages) / num_proc)
	chunks = [
		pages[i:i + chunk_size] 
		for i in range(0, len(pages), chunk_size)
	]

	# Define the arguments list.
	arg_list = [(args, device, file, chunk) for chunk in chunks]

	# Distribute the arguments among the pool of processes.
	with mp.Pool(processes=num_proc) as pool:
		# Aggregate the results of processes.
		results = pool.starmap(process_articles, arg_list)

		# Pass the aggregate results tuple to be merged.
		word_to_doc, doc_to_word, chunk_to_doc = merge_mappings(
			results
		)

	# Return the different mappings.
	return word_to_doc, doc_to_word, chunk_to_doc


def process_articles(args: Namespace, device: str, file: str, pages_str: List[str]):
	'''
	Preprocess the text (in a single thread/process).
	@param: args (Namespace), the arguments passed in from the 
		terminal.
	@param: device (str), the name of the CPU or GPU device that the
		embedding model will use.
	@param: file (str), the filepath of the current file being
		processed.
	@param: pages (List[str]), the raw xml text that is going to be
		processed.
	@return: returns the set of dictionaries containing the necessary 
		data and metadata to index the articles.
	'''
	# Initialize local mappings.
	word_to_doc = dict()
	doc_to_word = dict()
	chunk_to_doc = dict()

	# Pass each page string into beautifulsoup.
	pages = [BeautifulSoup(page, "lxml") for page in pages_str]

	# Load the configurations from the config JSON.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load the model tokenizer and model (if applicable). Do this here
	# instead of within the for loop for (runtime) efficiency.
	if args.vector:
		# Load the tokenizer and model.
		tokenizer, model = load_model(config)
	else:
		# Initialize variables to None
		tokenizer, model = None, None

	# for page in pages:
	for page in tqdm(pages):
		# Isolate the article/page's SHA1.
		sha1_tag = page.find("sha1")

		if sha1_tag is None:
			continue

		article_sha1 = sha1_tag.get_text()
		# print(f"\tArticle SHA1 {article_sha1}")

		# Isolate the article/page's raw text.
		article_text = process_page(page)
		article_text_bow = copy.deepcopy(article_text)
		article_text_v_db = copy.deepcopy(article_text)

		###############################################################
		# BAG OF WORDS
		###############################################################
		if args.bow:
			# Create a bag of words for each article (xml) file.
			xml_bow, xml_word_freq = bow_preprocessing(
				article_text_bow, return_word_freq=True
			)

			# Update word to document map.
			file_hash = file + article_sha1
			for word in xml_bow:
				if word in list(word_to_doc.keys()):
					# word_to_doc[word].append(file)
					word_to_doc[word].append(file_hash)
				else:
					# word_to_doc[word] = [file]
					word_to_doc[word] = [file_hash]

			# Update the document to words map.
			# doc_to_word[file] = xml_word_freq
			doc_to_word[file_hash] = xml_word_freq

		###############################################################
		# VECTOR EMBEDDINGS
		###############################################################
		if args.vector:
			# Assertion to make sure tokenizer and model is
			# initialized.
			assert None not in [tokenizer, model], "Model tokenizer and model is expected to be initialized for vector embeddings preprocessing."

			# Pass the article 
			xml_chunks = vector_preprocessing(
				article_text_v_db, config, tokenizer
			)
			pass

			# Embed chunks and write them to vector storage.
			# for chunk in xml_chunks:
			# 	pass
	
	# Return the mappings.
	return doc_to_word, word_to_doc, chunk_to_doc


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
	# PROGRAM ARGUMENTS
	###################################################################
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--restart",
		action="store_true",
		help="Specify whether to restart the preprocessing from scratch. Default is false/not specified."
	)
	parser.add_argument(
		"--multi_proc",
		action="store_true",
		help="Specify whether to use multiprocessing in preprocessing. Default is false/not specified."
	)
	parser.add_argument(
		"--bow",
		action="store_true",
		help="Specify whether to run the bag-of-words preprocessing. Default is false/not specified."
	)
	parser.add_argument(
		"--vector",
		action="store_true",
		help="Specify whether to run the vector database preprocessing. Default is false/not specified."
	)
	args = parser.parse_args()

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

	# NOTE:
	# Be careful if you are on mac. Because Apple Silicon works off of
	# the unified memory model, there may be some performance hit for 
	# CPU bound tasks. The hope is that MPS will actually accelerate 
	# the embedding model's performance.

	# GPU setup.
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"

	# Iterate through each file and preprocess it.
	for idx, file in enumerate(data_files):
		# Read in the file.
		with open(file, "r") as f:
			raw_text = f.read()

		print(f"Processing file {file}...")

		# Load the raw text into a beautifulsoup object and extract the
		# <page> tags.
		soup = BeautifulSoup(raw_text, "lxml")
		pages = soup.find_all("page")

		# NOTE:
		# We convert the list of <page> tags back to str because the
		# multiprocessing module is not able to handle the Tag | 
		# NavigableString objects as arguments to its starmap()
		# function. Even though it may come at a compute cost (some
		# additional latency), keeps the multiprocessing working which
		# helps speed up the whole process.

		# Convert each <page> tag in the list of pages to string.
		pages_str = [str(page) for page in pages]

		# NOTE:
		# The tqdm package should come with one of the other packages
		# required for this project. Specifically, it should come with
		# either the pytorch or huggingface transformers packages. It
		# is useful for tracking the number of articles/pages that have
		# been processed in the file. Don't worry about bringing this
		# over the JS implementation.

		# TODO:
		# Investigate using multiprocessing to speed up the process.
		# There are hundreds of files to process and each one takes a
		# significant amount of time to comb through and preprocess.
		# The memory overhead is minimal, so single threaded/process
		# approach is still fine for underpowered or consumer 
		# computers. I just want the option for people who have more
		# powerful hardware and want results quicker.

		if args.multi_proc:
			print('enabling multi processing')

			# Determine the number of CPU cores to use (this will be
			# passed down the the multiprocessing function)
			max_proc = min(mp.cpu_count(), 32)

			# Reset the device if the number of processes to be used is
			# greater than 4. This is because the device setting is
			# quite rudimentary with this system. I don't know
			# 1) How much VRAM each instance of a model would take up 
			#	vs the amount of VRAM available (4Gb, 8Gb, 12GB, ...).
			# 2) How transformers or pytorch would have to be 
			#	configured to balance the number of model instances on
			#	each process against multiple GPUs on device.
			# For now, this just makes it simpler.
			if max_proc > 4:
				device = "cpu"

			multiprocess_articles(args, device, file, pages_str, num_proc=max_proc)
		else:
			print("enabling serial processing")
			process_articles(args, device, file, pages_str)


		# Compute size of mappings.
		# d2w_size = sys.getsizeof(doc_to_word)
		# w2d_size = sys.getsizeof(word_to_doc)

		# GB_SIZE = 1024 * 1024 * 1024
		# if d2w_size // GB_SIZE > 1:
		# 	print(f"Document to word map has reached over 1GB in size")
		# 	exit()
		# elif w2d_size // GB_SIZE > 1:
		# 	print(f"Word to document map has reached over 1GB in size")
		# 	exit()
		exit()


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