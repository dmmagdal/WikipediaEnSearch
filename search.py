# search.py
# Implement search methods on the dowloaded (preprocessed) wikipedia
# data.
# Python 3.9
# Windows/MacOS/Linux


import concurrent.futures
import copy
import gc
import hashlib
import heapq
import json
import math
import os
import string
import time
from typing import List, Dict, Tuple

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
# import faiss
import lancedb
import msgpack
import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from preprocess import load_model, process_page
from preprocess import bow_preprocessing, vector_preprocessing
from generate_trie import load_trie


def hashSum(data: str) -> str:
	'''
	Compute the SHA256SUM of the xml data. This is used as part of the
		naming scheme down the road.
	@param: data (str), the raw string data from the xml data.
	@return: returns the SHA256SUM hash.
	'''
	# Initialize the SHA256 hash object.
	sha256 = hashlib.sha256()

	# Update the hash object with the (xml) data.
	sha256.update(data.encode('utf-8'))

	# Return the digested hash object (string).
	return sha256.hexdigest()


def load_article_xml_file(path: str) -> str:
	'''
	Load an xml file from the given path.
	@param: path (str), the path of the xml file that is to be loaded.
	@return: Returns the xml file contents.
	'''
	with open(path, "r") as f:
		return f.read()


def load_article_text(path: str, sha1_list: List[str]) -> List[str]:
	'''
	Load the specified articles from an xml file given the path.
	@param: path (str), the path of the xml file that is to be loaded.
	@param: sha1_list (List[str]), a list the SHA1 hashes of the target 
		articles.
	@return: Returns a list containing the article text of each file.
	'''
	# Load the (xml) file.
	file = load_article_xml_file(path)

	# Parse the file with beautifulsoup.
	soup = BeautifulSoup(file, "lxml")

	# Initialize list with default return values for all SHA1s passed
	# into the SHA1 list argument.
	articles = [
		f"ERROR: COULD NOT LOCATE ARTICLE {sha1_hash} in {path}"
		for sha1_hash in sha1_list
	]

	# Iterate through every article/page in the file.
	for page in soup.find_all("page"):
		# Isolate the article/page's SHA1.
		sha1_tag = page.find("sha1")

		# Skip articles that don't have a SHA1 (should not be possible 
		# but you never know).
		if sha1_tag is None:
			continue

		# Clean article SHA1 text.
		article_sha1 = sha1_tag.get_text()
		article_sha1 = article_sha1.replace(" ", "").replace("\n", "")

		# If the article/page's SHA1 matches one of the values in the 
		# SHA1 list, load the text for that article in the appropriate
		# spot in the list.
		if article_sha1 in sha1_list:
			sha1_index = sha1_list.index(article_sha1)
			articles[sha1_index] = process_page(page)

	# NOTE
	# The above process is a linear operation. It can be optimized for
	# multithreading/processing. There is also no "smart" or clever way
	# to just isolate pages with the desired SHA1 hashes on account of 
	# the fact that the strings containing the hashes are dirty in the
	# original file (they contain white space and newline characters)
	# so a string match becomes a bit difficult.

	# # Initiaize the return list of articles.
	# articles = []

	# # Iterate through the different SHA1 values from the SHA1 hash
	# # list.
	# for sha1_hash in sha1_list:
	# 	# Isolate the target article with the given SHA1 hash. Append
	# 	# the processed text if it was found, otherwise append the
	# 	# error message string.
	# 	page = soup.find("page", attrs={"sha1": sha1_hash})
	# 	# page = soup.find("page", sha1=sha1_hash)
	# 	if page is not None:
	# 		articles.append(process_page(page))
	# 	else:
	# 		articles.append(f"ERROR: COULD NOT LOCATE ARTICLE {sha1_hash} in {path}")

	# Return the list of article texts.
	return articles


def load_data_from_msgpack(path: str) -> Dict:
	'''
	Load a data file (to dictionary) from msgpack file given the path.
	@param: path (str), the path of the data file that is to be loaded.
	@return: Returns a python dictionary containing the structured data
		from the loaded data file.
	'''
	with open(path, 'rb') as f:
		byte_data = f.read()

	return msgpack.unpackb(byte_data)


def load_data_from_json(path: str) -> Dict:
	'''
	Load a data file (to dictionary) from either a file given the path.
	@param: path (str), the path of the data file that is to be loaded.
	@return: Returns a python dictionary containing the structured data
		from the loaded data file.
	'''
	with open(path, "r") as f:
		return json.load(f)
	

def load_data_file(path: str, use_json: bool = False) -> Dict:
	'''
	Load a data file (to dictionary) from either a JSON or msgpack file
		given the path.
	@param: path (str), the path of the data file that is to be loaded.
	@param: use_json (bool), whether to load the data file using JSON 
		msgpack (default is False).
	@return: Returns a python dictionary containing the structured data
		from the loaded data file.
	'''
	if use_json:
		return load_data_from_json(path)
	return load_data_from_msgpack(path)


def cosine_similarity(vec1: List[float], vec2: List[float]):
	'''
	Compute the cosine similarity of two vectors.
	@param: vec1 (List[float]), a vector of float values. This can be
		vectors such as a sparse vector from TF-IDF or BM25 to a dense
		vector like an embedding vector.
	@param: vec2 (List[float]), a vector of float values. This can be
		vectors such as a sparse vector from TF-IDF or BM25 to a dense
		vector like an embedding vector.
	@return: Returns the cosine similarity between the two input 
		vectors. Value range is 0 (similar) to 1 (disimilar).
	'''
	# Convert the vectors to numpy arrays.
	np_vec1 = np.array(vec1)
	np_vec2 = np.array(vec2)

	# Compute the cosine similarity of the two vectors and return the
	# value.
	cosine = np.dot(np_vec1, np_vec2) /\
		(np.linalg.norm(np_vec1) * np.linalg.norm(np_vec2))
	return cosine
	

def print_results(results: List, search_type: str = "tf-idf") -> None:
	valid_search_types = ["tf-idf", "bm25", "vector", "rerank"]
	assert search_type in valid_search_types,\
		f"Expected 'search_type' to be either {', '.join(valid_search_types)}. Received {search_type}"

	# Format of the input search results:
	# TF-IDF
	# [cosine similarity, document (file + SHA1), text, indices]
	# BM25
	# [score, document (file + SHA1), text, indices]
	# Vector
	# [cosine similarity, document (file + SHA1), text, indices]
	# ReRank
	# [cosine similarity, document (file + SHA1), text, indices]

	print(f"SEARCH RESULTS:")
	print('-' * 72)
	for result in results:
		# Deconstruct the results.
		score, document, text, indices = result
		doc, sha1 = os.path.basename(document).split(".xml")
		doc += ".xml"

		# Print the results out.
		print(f"Score: {score}")
		print(f"File Path: {doc}")
		print(f"Article SHA1: {sha1}")
		print(f"Article Text:\n{text[indices[0]:indices[1]]}")


class BagOfWords: 
	def __init__(self, bow_dir: str, corpus_size: int=-1, srt: float=-1.0, use_json=False) -> None:
		'''
		Initialize a Bag-of-Words search object.
		@param: bow_dir (str), a path to the directory containing the 
			bag of words metadata. This metadata includes the folders
			mapping to the word-to-document and document-to-word files.
		@param: srt (float), the sparse retrieval threshold. A value
			used to remove documents from the search results if they
			score a cosine similarity above the threshold.
		'''
		# Initialize class variables either from arguments or with
		# default values.
		self.bow_dir = bow_dir
		self.word_to_doc_folder = None
		self.doc_to_word_folder = None
		self.word_to_doc_files = None
		self.doc_to_word_files = None
		self.idf_files = None
		self.int_to_doc_file = None
		self.documents_folder = "./WikipediaEnDownload/WikipediaData"
		self.corpus_size = corpus_size	# total number of documents (articles)
		self.srt = srt					# similarity relative threshold value
		self.use_json = use_json
		self.extension = ".json" if use_json else ".msgpack"
		self.alpha_numerics = string.digits + string.ascii_lowercase
		self.word_len_limit = 60
		self.idf_threshold = 1.0

		# Initialize mapping folder path and files list.
		self.locate_and_validate_documents(bow_dir)

		# Verify that the class variables that were initialized as None
		# by default are no longer None.
		initialized_variables = [
			self.word_to_doc_folder, self.doc_to_word_folder,
			self.word_to_doc_files, self.doc_to_word_files,
			self.idf_files, self.int_to_doc_file,
		]
		assert None not in initialized_variables,\
			"Some variables were not initialized properly"
		
		# Compute the corpus size if the default value for the argument 
		# is detected.
		if self.corpus_size == -1:
			self.corpus_size = self.get_number_of_documents()

		# Verify that the corpus size is not 0.
		assert self.corpus_size != 0,\
			"Could not count the number of documents (articles) in the corpus. Corpus size is 0."
		
		# Verify that the srt is either -1.0 or in the range [0.0, 1.0]
		# (cosine similarity range).
		srt_off = self.srt == -1.0
		srt_valid = self.srt >= 0.0 and self.srt <= 1.0
		assert srt_off or srt_valid,\
			"SRT value was initialize to an invalid number. Either -1.0 for 'off' or a float in the range [0.0, 1.0] is expected"


	def locate_and_validate_documents(self, bow_dir: str):
		'''
		Verify that the bag-of-words directory exists along with the
			metadata files expected to be within them.
		@param: bow_dir (str), a path to the directory containing the 
			bag of words metadata. This metadata includes the folders
			mapping to the word-to-document and document-to-word files.
		@return: returns nothing.
		'''
		# Initialize path to word to document and document to word 
		# folders.
		self.word_to_doc_folder = os.path.join(bow_dir, 'word_to_docs')
		self.doc_to_word_folder = os.path.join(bow_dir, 'doc_to_words')

		# Initialize path to word to idf and trie folders (trie folder
		# contains document id to document mapping as well).
		self.idf_folder = os.path.join(bow_dir, "idf_cache")
		self.trie_folder = os.path.join(bow_dir, "trie_cache")
		self.redirect_folder = os.path.join(bow_dir, "redirect_cache")

		# Verify that the paths exist.
		assert os.path.exists(self.word_to_doc_folder) and os.path.isdir(self.word_to_doc_folder),\
			f"Expected path to word to documents folder to exist: {self.word_to_doc_folder}"
		assert os.path.exists(self.doc_to_word_folder) and os.path.isdir(self.doc_to_word_folder),\
			f"Expected path to document to words folder to exist: {self.doc_to_word_folder}"
		
		# Initialize the list of files for each mapping folder.
		self.word_to_doc_files = [
			os.path.join(self.word_to_doc_folder, file)
			for file in os.listdir(self.word_to_doc_folder)
			if file.endswith(self.extension)
		]
		self.doc_to_word_files = [
			os.path.join(self.doc_to_word_folder, file)
			for file in os.listdir(self.doc_to_word_folder)
			if file.endswith(self.extension)
		]

		# Initialize path to word to idf mappings, trie files, and 
		# document id to document mappings.
		doc_id_map_files = [
			"doc_to_int" + self.extension,
			"int_to_doc" + self.extension,
		]
		self.idf_files = [
			os.path.join(self.idf_folder, file)
			for file in os.listdir(self.idf_folder)
			if file.endswith(self.extension)
		]
		self.int_to_doc_file = os.path.join(
			self.trie_folder, doc_id_map_files[1]
		)
		self.redirect_files = [
			os.path.join(self.redirect_folder, file)
			for file in os.listdir(self.redirect_folder)
			if file.endswith(self.extension)
		]
		
		# Verify that the list of files for each mapping folder is not
		# empty.
		assert len(self.word_to_doc_files) != 0,\
			f"Detected word to documents folder {self.word_to_doc_folder} does have not supported files"
		assert len(self.word_to_doc_files) != 0,\
			f"Detected document to words folder {self.doc_to_word_folder} does have not supported files"
		assert len(self.idf_files) != 0,\
			f"Detected word to idf folder {self.idf_folder} does have not supported files"
		assert os.path.exists(self.int_to_doc_file),\
			f"Required document id to document file in {self.trie_folder} does exist"
		assert len(self.redirect_files) != 0,\
			f"Detected redirected documents folder {self.redirect_folder} does have not supported files"


	def get_number_of_documents(self) -> int:
		'''
		Count the number of documents recorded in the corpus.
		@param, takes no arguments.
		@return, Returns the number of documents in the corpus.
		'''
		# Initialize the counter to 0.
		counter = 0

		# Iterate through each file in the documents to words map 
		# files.
		print("Getting the number of documents in the corpus...")
		for file in tqdm(self.doc_to_word_files):
			# Load the data from the file and increment the counter by
			# the number of documents in each file.
			doc_to_words = load_data_file(file, self.use_json)
			counter += len(list(doc_to_words.keys()))
		
		# Return the count.
		return counter
	

	def inverted_index(self, words: List[str], index: str = "trie") -> List[str]:
		'''
		Pass query words to the inverted index to retrieve the list of 
			documents to search from.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to search.
		@param: index (str), the type of inverted index to use.
		@return: returns a list of unique relevant documents based on 
			the inverted index used.
		'''
		valid_inverted_indices = [
			"trie", "weighted_doc_idf", "word_idf_filter", 
			"category_word2vec", "category_weighted_tfidf"
		]
		assert index in valid_inverted_indices, \
			f"Inverted index type {index} did not match list of valid types: {', '.join(valid_inverted_indices)}"
		
		# Return the documents from the inverted index.
		if index == "trie":
			# Pass the words list ot the plain inverted index
			return self.get_documents_from_trie(words)
		elif index == "weighted_doc_idf":
			return self.get_documents_from_trie_weighted(words)
		elif index == "word_idf_filter":
			# Remove words from the query based on the filter.
			filtered_words = [
				word for word in words
				if self.compute_idf(word)[0] < self.idf_threshold
			]

			# Pass the filtered words list to the plain inverted index.
			return self.get_documents_from_trie(filtered_words)
		elif index == "category_word2vec":
			pass
		elif index == "ategory_weighted_tfidf":
			pass

		return []
	

	def get_documents_from_trie(self, words: List[str]) -> List[str]:
		'''
		Query the trie inverted index given the query to retrieve the 
			list of documents to search from.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to search.
		@return: returns a list of unique relevant documents based on 
			the inverted index used.
		'''
		# Initialize a dictionary to group words together by their
		# first character.
		char_word_dict = dict()

		# Load the document id to documents mappings.
		int_to_doc = load_data_file(
			self.int_to_doc_file, self.use_json
		)

		# Convert key back to int for document id to documents map.
		int_to_doc = {
			int(key): value for key, value in int_to_doc.items()
		}

		# Iterate through each word in the query words list.
		for word in words:
			# Isolate the first character in the word.
			word_char = word[0]

			# Reset the char variable if it is not an alphanumeric.
			if word_char not in self.alpha_numerics:
				word_char = "other"

			# Verify that the word is within the set word length limit.
			# Will skip the word otherwise.
			if len(word) <= self.word_len_limit:
				# Store the word to the character to word dictionary.
				if word_char in list(char_word_dict.keys()):
					char_word_dict[word_char].append(word)
				else:
					char_word_dict[word_char] = [word]

		# Initialize the set of documents that will be returned. Each
		# item in the list will be a unique string.
		documents = set()

		# Iterate through the characters in the character to word
		# dictionary.
		for char in sorted(list(char_word_dict.keys())):
			# Reset the char variable if it is not an alphanumeric.
			if char not in self.alpha_numerics:
				char = "other"

			# Unpack the words in the character to words dictionary.
			char_words = char_word_dict[char]

			# Identify the list of trie shards for this character.
			shard_files = [
				shard_path 
				for shard_path in list(self.trie_shard_map.keys())
				if os.path.basename(shard_path).startswith(f"{char}_shard") and \
					shard_path.endswith(self.extension)
			]

			# Initialize a map of character trie shards to words.
			shard_to_words = dict()

			# Map each character trie shard to a word if applicable.
			for shard in shard_files:
				word_range = self.trie_shard_map[shard]
				for idx, word in enumerate(char_words):
					# From generate_trie.py: words were sorted 
					# lexicographically. Means that this is how the
					# checking the range should look 
					# (small <= word <= large).
					if word_range[0] <= word <= word_range[-1]:
						if shard in list(shard_to_words.keys()):
							shard_to_words[shard].append(word)
						else:
							shard_to_words[shard] = [word]

			# Iterate through all character trie shards that map to at
			# least one word.
			for shard in list(shard_to_words.keys()):
				# Load the words that map to the trie shard.
				print(f"Loading from trie shard {shard}")
				search_words = shard_to_words[shard]

				# Load the trie shard.
				start = time.perf_counter()
				trie = load_trie(shard, self.use_json)
				end = time.perf_counter()
				print(f"Time to load shard: {(end - start):.6f} seconds")

				# Iterate through each word. Update the documents set
				# if the results returned from the trie are valid (not
				# None).
				for word in search_words:
					results = trie.search(word)
					if results is not None:
						documents.update([
							int_to_doc[result] for result in results
						])

				# Memory cleanup.
				del trie
				gc.collect()

			# Memory cleanup.
			del shard_files
			del shard_to_words
			gc.collect()

		# Convert the documents set to a list and return it.
		return list(documents)
	

	def get_documents_from_trie_weighted(self, words: List[str]) -> List[str]:
		'''
		Query the trie inverted index given the query to retrieve the 
			list of documents to search from. Documents are filtered 
			based on their cumulative weights from the word IDFs.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to search.
		@return: returns a list of unique relevant documents based on 
			the inverted index used.
		'''
		# Initialize a dictionary to group words together by their
		# first character.
		char_word_dict = dict()

		# Load the document id to documents mappings. Convert key back 
		# to int for document id to documents map.
		int_to_doc = load_data_file(
			self.int_to_doc_file, self.use_json
		)
		int_to_doc = {
			int(key): value for key, value in int_to_doc.items()
		}

		# Iterate through each word in the query words list.
		for word in words:
			# Isolate the first character in the word.
			word_char = word[0]

			# Reset the char variable if it is not an alphanumeric.
			if word_char not in self.alpha_numerics:
				word_char = "other"

			# Verify that the word is within the set word length limit.
			# Will skip the word otherwise.
			if len(word) <= self.word_len_limit:
				# Store the word to the character to word dictionary.
				if word_char in list(char_word_dict.keys()):
					char_word_dict[word_char].append(word)
				else:
					char_word_dict[word_char] = [word]

		# Initialize the mapping of documents to their respective IDF
		# weights.
		doc_to_idf = dict()

		# Iterate through the characters in the character to word
		# dictionary.
		for char in sorted(list(char_word_dict.keys())):
			# Reset the char variable if it is not an alphanumeric.
			if char not in self.alpha_numerics:
				char = "other"

			# Unpack the words in the character to words dictionary.
			char_words = char_word_dict[char]
			idf = self.compute_idf(char_words)

			# Identify the list of trie shards for this character.
			shard_files = [
				shard_path 
				for shard_path in list(self.trie_shard_map.keys())
				if os.path.basename(shard_path).startswith(f"{char}_shard") and \
					shard_path.endswith(self.extension)
			]

			# Initialize a map of character trie shards to words.
			shard_to_words = dict()

			# Map each character trie shard to a word if applicable.
			for shard in shard_files:
				word_range = self.trie_shard_map[shard]
				for idx, word in enumerate(char_words):
					# From generate_trie.py: words were sorted 
					# lexicographically. Means that this is how the
					# checking the range should look 
					# (small <= word <= large).
					if word_range[0] <= word <= word_range[-1]:
						if shard in list(shard_to_words.keys()):
							shard_to_words[shard].append(word)
						else:
							shard_to_words[shard] = [word]

			# Iterate through all character trie shards that map to at
			# least one word.
			for shard in list(shard_to_words.keys()):
				# Load the words that map to the trie shard.
				print(f"Loading from trie shard {shard}")
				search_words = shard_to_words[shard]

				# Load the trie shard.
				trie = load_trie(shard, self.use_json)

				# Iterate through each word. Update the document to IDF
				# weight mappings if the results returned from the trie
				# are valid (not None).
				for word in search_words:
					results = trie.search(word)
					if results is not None:
						for result in results:
							document = int_to_doc[result]
							idf_value = idf[char_words.index(word)]
							if document in list(doc_to_idf.keys()):
								doc_to_idf[document] += idf_value
							else:
								doc_to_idf[document] = idf_value

				# Memory cleanup.
				del trie
				gc.collect()

			# Memory cleanup.
			del shard_files
			del shard_to_words
			gc.collect()

		# Filter out documents with an IDF weight below the set 
		# threshold. Return the list of documents.
		documents = [
			doc for doc, idf in doc_to_idf.items() 
			if idf > self.idf_threshold
		]
		return documents


	def get_document_paths_from_documents(self, documents: List[str]) -> Dict[str, List[str]]:
		'''
		Given a document list (each document is a file + article hash),
			return a dictionary mapping the full path of all unique 
			files to the expected article hashes within each file.
		@param: documents (List[str]), the list of all documents that
			were returned by the inverted index.
		@return: returns a dictionary mapping all files to the 
			respective article hashes within each file.
		'''
		# Initialize dictionary mapping each file to the list of 
		# expected hashes.
		file_article_dict = dict()

		# Isolate directory path and basename of file.
		basenames = [
			(os.path.dirname(doc), os.path.basename(doc))
			for doc in documents
		]

		# for folder, name in basenames:
		for _, name in basenames:
			file_basename, article_hash = name.split(".xml")
			file_basename += ".xml"
			# file = os.path.join(folder, file_basename)
			file = file_basename

			if file in list(file_article_dict.keys()):
				file_article_dict[file].append(article_hash)
			else:
				file_article_dict[file] = [article_hash]

		return file_article_dict
	

	def compute_tf(self, doc_word_freq: Dict, words: List[str]) -> List[float]:
		'''
		Compute the Term Frequency of a set of words given a document's
			word frequency mapping.
		@param: doc_word_freq (Dict), the mapping of a given word the
			frequency it appears in a given document.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@return: returns the Term Frequency of each of the words for
			the given document in a vector (List[float]). The vector is
			ordered such that the index of each value corresponds to 
			the index of a word in the word list argument.
		'''
		# Initialize the document's term frequency vector.
		doc_word_tf = [0.0] * len(words)

		# Compute total word count.
		total_word_count = sum(
			[value for value in doc_word_freq.values()]
		)

		# Compute the term frequency accordingly and add it to the 
		# document's word vector
		for word_idx in range(len(words)):
			word = words[word_idx]
			if word in doc_word_freq:
				word_freq = doc_word_freq[word]
				doc_word_tf[word_idx] =  word_freq / total_word_count
			
		# Return the document's term frequency for input words as a 
		# vector (List[float]).
		return doc_word_tf
	

	def compute_idf(self, words: List[str]) -> List[float]:
		'''
		Retrieve the precomputed Inverse Document Frquency of the given
		 	set of (usually query) words.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: returns the Inverse Document Frequency for all words
			queried in the corpus. The data is returned in an ordered
			list (List[float]) where the index of each value
			corresponds to the index of a word in the word list 
			argument.
		'''
		# Initialize a list containing the mappings of the query words
		# to the IDF.
		idf_vector = [0.0] * len(words)

		# Iterate through each file.
		for file in self.idf_files:
			# Load the word to IDF mappings from file.
			word_to_idf = load_data_file(file, use_json=self.use_json)

			# Iterate through each word and retrieve the IDF value for
			# that word if it is available.
			for word_idx in range(len(words)):
				word = words[word_idx]
				if word in word_to_idf:
					idf_vector[word_idx] = word_to_idf[word]

		# Return the inverse document frequency vector.
		return idf_vector


class TF_IDF(BagOfWords):
	def __init__(self, bow_dir: str, corpus_size: int=-1, srt: float=-1.0, use_json=False) -> None:
		super().__init__(
			bow_dir=bow_dir, corpus_size=corpus_size, srt=srt, 
			use_json=use_json
		)
		pass


	def search(self, query: str, max_results: int = 50):
		'''
		Conducts a search on the wikipedia data with TF-IDF.
		@param: query str, the raw text that is being queried from the
			wikipedia data.
		@param: max_results (int), the maximum number of search results
			to return.
		@return: returns a list of objects where each object contains
			an article's path, title, retrieved text, and the slices of
			that text that is being returned (for BM25 and TF-IDF, 
			those slices values are for the whole article).
		'''
		# Assertion for max_results argument (must be non-zero int).
		assert isinstance(max_results, int) and str(max_results).isdigit() and max_results > 0, f"max_results argument is expected to be some int value greater than zero. Recieved {max_results}"

		# Preprocess the search query to a bag of words.
		words, word_freq = bow_preprocessing(query, True)

		# Isolate a list of files/documents to look through.
		words = sorted(words)

		# num_workers used:
		# 8 -> 
		# 16 -> 
		# 32 -> OOM on macbook (capped at around 15 GB -> includes swap/cache)
		# NOTE:
		# Number of threads seems to be gated by hardware (memory). The
		# largest trie bundle is 417MB and 389MB for the doc to words
		# mapping. This means that for any thread, the memory overhead 
		# is at around 1GB. Fewer threads means less 
		# parallelization/concurrency BUT since this is a hardware 
		# constraint, I've already pushed optimization as far as I can
		# (with the exception of converting to rust).
		# TODO:
		# Add num_workers as an __init__() argument for the BagOfWords
		# class and set it in the config.json.

		num_workers = 8
		chunk_size = math.ceil(len(self.doc_to_word_files) / num_workers)
		file_chunks = [
			self.doc_to_word_files[i:i + chunk_size]
			for i in range(9, len(self.doc_to_word_files), chunk_size)
		]

		word_idfs = self.compute_idf(words)
		print("Captured word IDFs")
		args_list = [
			(file_chunk, words, word_freq, word_idfs, max_results)
			for file_chunk in file_chunks
		]
		corpus_tfidf = []
		with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
			print("Starting multithreaded processing")
			results = executor.map(lambda args: self.file_search(*args), args_list)
			
			for result in results:
				while len(result) > 0:
					result_item = result.pop()
					if result_item in corpus_tfidf:
						continue
					if max_results != -1 and len(corpus_tfidf) >= max_results:
						# Pushpop the highest (cosine similarity) value
						# tuple from the heap to make way for the next
						# tuple.
						heapq.heappushpop(
							corpus_tfidf,
							result_item
						)
					else:
						heapq.heappush(
							corpus_tfidf,
							result_item
						)

		# target_documents = self.get_documents_from_trie(words)
		# # target_documents = self.inverted_index(words, "word_idf_filter")
		# print("Isolated documents from tries.")
		# print(json.dumps(target_documents[:10], indent=4))
		# print(len(target_documents))

		# # Compute the TF-IDF for the corpus.
		# # _, corpus_tfidf = self.compute_tfidf(
		# # 	words, word_freq, max_results=max_results
		# # )
		# corpus_tfidf = self.compute_tfidf(
		# 	words, word_freq, target_documents, max_results=max_results
		# )

		# The corpus TF-IDF results are stored in a max heap. Convert
		# the structure back to a list sorted from smallest to largest
		# cosine similarity score.
		sorted_rankings = []
		for _ in range(len(corpus_tfidf)):
			# Pop the top item from the max heap.
			result = heapq.heappop(corpus_tfidf)

			# Reverse the cosine similarity score back to its original
			# value.
			result[0] *= -1

			# Extract the document path and SHA1 and use them to load 
			# the article text.
			document_sha1 = result[1]
			document, sha1 = os.path.basename(document_sha1).split(".xml")
			document = os.path.join(self.documents_folder, document + ".xml")
			text = load_article_text(document, [sha1])[0]
			
			# Append the results.
			full_result = result + [text, [0, len(text)]]

			# Insert the item into the list from the front.
			sorted_rankings.insert(0, full_result)

		# Return the list.
		return sorted_rankings
	

	def file_search(self, doc_to_word_files, words, word_freq, word_idfs, max_results):
		basenames = [
			os.path.basename(file).replace(self.extension, "") 
			for file in doc_to_word_files
		]
		trie_paths = [
			os.path.join(self.trie_folder, folder) 
			for folder in os.listdir(self.trie_folder)
			if os.path.isdir(os.path.join(self.trie_folder, folder)) and folder in basenames
		]
		assert len(basenames) == len(trie_paths)

		# Stack heap for the search.
		stack_heap = list()
		heapq.heapify(stack_heap)

		for trie_path in trie_paths:
			basename = os.path.basename(trie_path)

			###########################################################
			# Inverted Index Search
			###########################################################

			# Initialize the set of documents that will be returned. 
			# Each item in the list will be a unique string.
			local_documents = set()

			# Isolate document mapping files.
			local_int_to_doc_path = os.path.join(
				trie_path, f"int_to_doc{self.extension}"
			)
			int_to_doc = load_data_file(
				local_int_to_doc_path, self.use_json
			)

			# Convert key back to int for document id to documents map.
			int_to_doc = {
				int(key): value for key, value in int_to_doc.items()
			}

			# Initialize a dictionary to group words together by their
			# first character.
			char_word_dict = dict()

			# Iterate through each word in the query words list.
			for word in words:
				# Isolate the first character in the word.
				word_char = word[0]

				# Reset the char variable if it is not an alphanumeric.
				if word_char not in self.alpha_numerics:
					word_char = "other"

				# Verify that the word is within the set word length limit.
				# Will skip the word otherwise.
				if len(word) <= self.word_len_limit:
					# Store the word to the character to word dictionary.
					if word_char in list(char_word_dict.keys()):
						char_word_dict[word_char].append(word)
					else:
						char_word_dict[word_char] = [word]

			# Iterate through the characters in the character to word
			# dictionary.
			for char in sorted(list(char_word_dict.keys())):
				# Reset the char variable if it is not an alphanumeric.
				if char not in self.alpha_numerics:
					char = "other"

				# Unpack the words in the character to words dictionary.
				char_words = char_word_dict[char]

				# Identify the trie for this character.
				trie_file = os.path.join(
					trie_path, f"{char}_trie_slim{self.extension}"
				)
				assert os.path.exists(trie_file), f"Trie file {trie_file} was expected but not found."

				# Load the trie.
				# start = time.perf_counter()
				trie = load_trie(trie_file, self.use_json)
				# end = time.perf_counter()
				# print(f"Time to load shard: {(end - start):.6f} seconds")

				# Iterate through each word. Update the documents set
				# if the results returned from the trie are valid (not
				# None).
				for word in char_words:
					results = trie.search(word)
					if results is not None:
						local_documents.update([
							int_to_doc[result] for result in results
						])
	
				# Memory cleanup.
				del trie
				gc.collect()

			# Convert the documents set to a list.
			local_documents = list(local_documents)
			print(f"Local documents: {len(local_documents)}")

			###########################################################
			# TF-IDF Ranking
			###########################################################

			# Compute the TF-IDF for the file.
			# file_tfidf = self.compute_tfidf(
			# 	words, word_freq, word_idf, 
			# 	local_documents, max_results=max_results
			# )

			# Compute query TF-IDF.
			query_total_word_count = sum(
				[value for value in word_freq.values()]
			)
			query_tfidf = [0.0] * len(words)
			for word_idx in range(len(words)):
				word = words[word_idx]
				query_word_tf = word_freq[word] / query_total_word_count
				query_tfidf[word_idx] = query_word_tf * word_idfs[word_idx]

			# Load doc to word frequency mappings for the file.
			doc_to_words = load_data_file(
				os.path.join(self.doc_to_word_folder, 
				f"{basename}{self.extension}"),
				self.use_json
			)

			# Compute the intersection of the documents passed in from
			# arguments and the current list of documents in the file.
			document_intersect = set(local_documents).intersection(
				list(doc_to_words.keys())
			)
			print(f"Thread doc interstection {len(document_intersect)}")
			print(f"local and intersection match: {len(local_documents) == len(document_intersect)}")

			for doc in list(document_intersect):
			# for doc in local_documents:
				# Extract the document word frequencies.
				word_freq_map = doc_to_words[doc]

				# Compute the document's term frequency for each word.
				doc_word_tf = self.compute_tf(word_freq_map, words)

				# Compute document TF-IDF.
				doc_tfidf = [
					tf * idf 
					for tf, idf in list(zip(doc_word_tf, word_idfs))
				]

				# Compute cosine similarity against query TF-IDF and
				# the document TF-IDF.
				doc_cos_score = cosine_similarity(
					query_tfidf, doc_tfidf
				)

				# If the sparse retrieval threshold has been 
				# initialized, verify the document cosine similarity
				# score is within that threshold. Do not append
				# documents to the results list if they fall under the
				# threshold.
				if self.srt > 0.0 and doc_cos_score > self.srt:
					continue

				# Multiply score by -1 to get inverse score. This is
				# important since we are relying on a max heap.
				doc_cos_score *= -1
				
				# Insert the document name (includes file path & 
				# article SHA1), TF-IDF vector, and cosine similarity 
				# score (against the query TF-IDF vector) to the heapq.
				# The heapq sorts by the first value in the tuple so 
				# that is why the cosine similarity score is the first
				# item in the tuple.
				# if max_results != -1 and len(file_tfidf_heap) >= max_results:
				if max_results != -1 and len(stack_heap) >= max_results:
					# Pushpop the highest (cosine similarity) value
					# tuple from the heap to make way for the next
					# tuple.
					heapq.heappushpop(
						stack_heap,
						# file_tfidf_heap,
						# tuple([doc_cos_score, doc, doc_tfidf])
						# [doc_cos_score, doc, doc_tfidf]
						[doc_cos_score, doc]
					)
				else:
					heapq.heappush(
						stack_heap,
						# file_tfidf_heap,
						# tuple([doc_cos_score, doc, doc_tfidf]) # Tuple doesnt support modification
						# [doc_cos_score, doc, doc_tfidf] # Results in issues unpacking list in preint_results()
						[doc_cos_score, doc]
					)

			# while len(file_tfidf_heap) > 0:
			# 	if max_results != -1 and len(stack_heap) >= max_results:
			# 		# Pushpop the highest (cosine similarity) value
			# 		# tuple from the heap to make way for the next
			# 		# tuple.
			# 		heapq.heappushpop(
			# 			stack_heap,
			# 			file_tfidf_heap.pop()
			# 		)
			# 	else:
			# 		heapq.heappush(
			# 			stack_heap,
			# 			file_tfidf_heap.pop()
			# 		)

		print(f"thread stack heap length: {len(stack_heap)}")
		return stack_heap
	

	def compute_tfidf(self, words: List[str], query_word_freq: Dict, documents: List[str], max_results: int = -1):
		'''
		Iterate through all the documents in the corpus and compute the
			TF-IDF for each document in the corpus. Sort the results
			based on the cosine similarity score and return the sorted
			list.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: query_word_freq (Dict), the word frequency mapping for 
			the input search query.
		@param: documents (List[str]), the list of files/documents that
			will be queried from the corpus.
		@param: max_results (int), the maximum number of results to 
			return. Default is -1 (no limit).
		@return: returns the sorted list of search results. 
		'''
		# Sort the set of words (ensures consistent positions of each 
		# word in vector).
		words = sorted(words)

		# Iterate through all files to get IDF of each query word. 
		# Compute only once per search.
		word_idf = self.compute_idf(words)

		# Compute query TF-IDF.
		query_total_word_count = sum(
			[value for value in query_word_freq.values()]
		)
		query_tfidf = [0.0] * len(words)
		for word_idx in range(len(words)):
			word = words[word_idx]
			query_word_tf = query_word_freq[word] / query_total_word_count
			query_tfidf[word_idx] = query_word_tf * word_idf[word_idx]

		# Compute corpus TF-IDF.
		corpus_tfidf_heap = []

		# Given the documents, get a filtered list of documents to use 
		# from doc_to_word_files.
		file_to_article = self.get_document_paths_from_documents(
			documents
		)
		filtered_files = [
			file 
			for file in self.doc_to_word_files
			if os.path.basename(file).replace(self.extension, ".xml") in list(file_to_article.keys())
		]
		# print("File to articles")
		# print(json.dumps(file_to_article, indent=4))
		# print(f"Matching files")
		# print(json.dumps(filtered_files, indent=4))

		# NOTE:
		# Heapq in use is a max-heap. This is implemented by 
		# multiplying the cosine similarity score by -1. That way, the
		# largest values are actually the smallest in the heap and are
		# popped when we need to pushpop the largest scoring tuple.
		print("Running TF-IDF search...")

		# Compute TF-IDF for every file.
		# for file in tqdm(self.doc_to_word_files):
		for file in tqdm(filtered_files):
			# Load the doc to word frequency mappings from file.
			doc_to_words = load_data_file(file, self.use_json)

			# Compute the intersection of the documents passed in from
			# arguments and the current list of documents in the file.
			document_intersect = set(documents).intersection(
				list(doc_to_words.keys())
			)

			# Iterate through each document.
			# for doc in doc_to_words:
			# Iterate through the documents provided from arguments.
			# for doc in documents:
			# Iterate through the document intersection.
			for doc in list(document_intersect):
				# Extract the document word frequencies.
				word_freq_map = doc_to_words[doc]

				# Compute the document's term frequency for each word.
				doc_word_tf = self.compute_tf(word_freq_map, words)

				# Compute document TF-IDF.
				doc_tfidf = [
					tf * idf 
					for tf, idf in list(zip(doc_word_tf, word_idf))
				]

				# Compute cosine similarity against query TF-IDF and
				# the document TF-IDF.
				doc_cos_score = cosine_similarity(
					query_tfidf, doc_tfidf
				)

				# If the sparse retrieval threshold has been 
				# initialized, verify the document cosine similarity
				# score is within that threshold. Do not append
				# documents to the results list if they fall under the
				# threshold.
				if self.srt > 0.0 and doc_cos_score > self.srt:
					continue

				# Multiply score by -1 to get inverse score. This is
				# important since we are relying on a max heap.
				doc_cos_score *= -1

				# NOTE:
				# Using heapq vs list keeps sorting costs down: 
				# list sort is n log n
				# list append is 1 or n (depending on if the list needs
				# to be resized)
				# heapify is n log n but since heap is initialized
				# from empty list, that cost is negligible
				# heapq pushpop is log n
				# heapq push is log n
				# heapq pop is log n
				# If shortening the list is a requirement, then I 
				# dont have to worry about sorting the list before
				# slicing it with heapq. The heapq will maintain
				# order with each operation at a cost efficent speed.
				
				# Insert the document name (includes file path & 
				# article SHA1), TF-IDF vector, and cosine similarity 
				# score (against the query TF-IDF vector) to the heapq.
				# The heapq sorts by the first value in the tuple so 
				# that is why the cosine similarity score is the first
				# item in the tuple.
				if max_results != -1 and len(corpus_tfidf_heap) >= max_results:
					# Pushpop the highest (cosine similarity) value
					# tuple from the heap to make way for the next
					# tuple.
					heapq.heappushpop(
						corpus_tfidf_heap,
						# tuple([doc_cos_score, doc, doc_tfidf])
						# [doc_cos_score, doc, doc_tfidf]
						[doc_cos_score, doc]
					)
				else:
					heapq.heappush(
						corpus_tfidf_heap,
						# tuple([doc_cos_score, doc, doc_tfidf]) # Tuple doesnt support modification
						# [doc_cos_score, doc, doc_tfidf] # Results in issues unpacking list in preint_results()
						[doc_cos_score, doc]
					)

		# Return the corpus TF-IDF.
		return corpus_tfidf_heap


class BM25(BagOfWords):#
	def __init__(self, bow_dir: str, k1: float = 1.0, b: float = 0.0, 
			  	corpus_size: int=-1, avg_doc_len: float=-1.0, srt: 
				float=-1.0, use_json=False) -> None:
		super().__init__(
			bow_dir=bow_dir, corpus_size=corpus_size, srt=srt, 
			use_json=use_json
		)
		self.avg_corpus_len = avg_doc_len
		if self.avg_corpus_len < 0.0:
			self.avg_corpus_len = self.compute_avg_corpus_size()
		self.k1 = k1
		self.b = b


	def compute_avg_corpus_size(self) -> float:
		'''
		Compute the average document length (in words) of the corpus.
		@param: takes no arguments
		@return: returns the average number of words per document 
			across the entire corpus.
		'''
		# Initialize document size sum.
		doc_size_sum = 0

		# Iterate through each file in the documents to words map 
		# files.
		print("Computing average document length of corpus...")
		for file in tqdm(self.doc_to_word_files):
			# Load the data from the file.
			doc_to_words = load_data_file(file, self.use_json)

			# For each document in the data, compute the length of the
			# document by adding up all the word frequency values.
			for doc in list(doc_to_words.keys()):
				doc_size_sum += sum(
					[value for value in doc_to_words[doc].values()]
				)

		# Return the average document size.
		return doc_size_sum / self.corpus_size


	def search(self, query: str, max_results: int = 50):
		'''
		Conducts a search on the wikipedia data with BM25.
		@param: query str, the raw text that is being queried from the
			wikipedia data.
		@param: max_results (int), the maximum number of search results
			to return.
		@return: returns a list of objects where each object contains
			an article's path, title, retrieved text, and the slices of
			that text that is being returned (for BM25 and TF-IDF, 
			those slices values are for the whole article).
		'''
		# Assertion for max_results argument (must be non-zero int).
		assert isinstance(max_results, int) and str(max_results).isdigit() and max_results > 0, f"max_results argument is expected to be some int value greater than zero. Recieved {max_results}"

		# Preprocess the search query to a bag of words.
		words = bow_preprocessing(query, False)
		words = words[0] # unpack return tuple.

		# Isolate a list of files/documents to look through.
		# target_documents = self.get_documents_from_trie(words)
		target_documents = self.inverted_index(words, index="word_idf_filter")

		# Compute the BM25 for the corpus.
		corpus_bm25 = self.compute_bm25(
			words, target_documents, max_results=max_results
		)

		# The corpus TF-IDF results are stored in a max heap. Convert
		# the structure back to a list sorted from largest to smallest 
		# BM25 score.
		sorted_rankings = []
		for _ in range(len(corpus_bm25)):
			# Pop the bottom item from the max heap.
			result = heapq.heappop(corpus_bm25)

			# Extract the document path and SHA1 and use them to load 
			# the article text.
			document_sha1 = result[1]
			document, sha1 = os.path.basename(document_sha1).split(".xml")
			document = os.path.join(self.documents_folder, document + ".xml")
			text = load_article_text(document, [sha1])[0]
			
			# Append the results.
			full_result = result + [text, [0, len(text)]]

			# Insert the item into the list from the end (append).
			sorted_rankings.append(full_result)
		
		# Return the list.
		return sorted_rankings


	def compute_bm25(self, words: List[str], documents: List[str], max_results: int = -1):
		'''
		Iterate through all the documents in the corpus and compute the
			BM25 for each document in the corpus. Sort the results
			based on the cosine similarity score and return the sorted
			list.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: query_word_freq (Dict), the word frequency mapping for 
			the input search query.
		@param: documents (List[str]), the list of files/documents that
			will be queried from the corpus.
		@param: max_results (int), the maximum number of results to 
			return. Default is -1 (no limit).
		@return: returns the BM25 for the query as well as the sorted
			list of search results. 
		'''
		# Sort the set of words (ensures consistent positions of each 
		# word in vector).
		words = sorted(words)

		# Iterate through all files to get IDF of each query word. 
		# Compute only once per search.
		word_idf = self.compute_idf(words)

		# Compute corpus BM25.
		corpus_bm25_heap = []

		# Given the documents, get a filtered list of documents to use 
		# from doc_to_word_files.
		file_to_article = self.get_document_paths_from_documents(
			documents
		)
		filtered_files = [
			file 
			for file in self.doc_to_word_files
			if os.path.basename(file).replace(self.extension, ".xml") in list(file_to_article.keys())
		]

		# NOTE:
		# Heapq in use is a max-heap. In this case, we don't want to 
		# multiply the BM25 score by -1 because a larger score means a
		# document is "more relevant" to the query (so we want to drop
		# the lower scores if we have a max_results limit). BM25 also 
		# doesn't require using cosine similarity since it aggregates
		# the term values into a sum for the document score.
		print("Running BM25 search...")

		# Compute BM25 for every file.
		# for file in tqdm(self.doc_to_word_files):
		for file in tqdm(filtered_files):
			# Load the doc to word frequency mappings from file.
			doc_to_words = load_data_file(file, self.use_json)

			document_intersect = set(documents).intersection(
				list(doc_to_words.keys())
			)

			# Iterate through each document.
			# for doc in doc_to_words:
			# for doc in documents:
			for doc in document_intersect:
				# Initialize the BM25 score for the document.
				bm25_score = 0.0

				# Extract the document word frequencies.
				word_freq_map = doc_to_words[doc]

				# Compute the document length.
				doc_len = sum(
					[value for value in word_freq_map.values()]
				)

				# Compute the document's term frequency for each word.
				doc_word_tf = self.compute_tf(word_freq_map, words)

				# Iterate over the different words and compute the BM25
				# score for each. Aggregate that score by adding it to 
				# the total BM25 score value.
				for word_idx in range(len(words)):
					tf = doc_word_tf[word_idx]
					numerator = word_idf[word_idx] * tf * (self.k1 + 1)
					denominator = tf + self.k1 *\
						(
							1 - self.b + self.b *\
							(doc_len / self.avg_corpus_len)
						)
					bm25_score += numerator / denominator

				# NOTE:
				# We ignore similarity relevance threshold here because
				# the range of values for BM25 scores are outside of
				# the range of values we've set for srt [0.0, 1.0].
				# Makes it a headache to deal with an unbounded range
				# so we'll make do without this optimization.

				# Insert the document name (includes file path & 
				# article SHA1), BM25 score to the heapq. The heapq 
				# sorts by the first value in the tuple so that is why
				# the cosine similarity score is the first item in the 
				# tuple.
				if max_results != -1 and len(corpus_bm25_heap) >= max_results:
					# Pushpop the smallest (BM25) value tuple from the 
					# heap to make way for the next tuple.
					heapq.heappushpop(
						corpus_bm25_heap,
						# tuple([bm25_score, doc])
						[bm25_score, doc]
					)
				else:
					heapq.heappush(
						corpus_bm25_heap,
						# tuple([bm25_score, doc])
						[bm25_score, doc]
					)

		# Return the corpus BM25 rankings.
		return corpus_bm25_heap


class VectorSearch:
	def __init__(self, model: str, index_dir: str, device: str = "cpu"):
		# Detect config.json file.
		assert os.path.exists("config.json"),\
			"Expected config.json file to exist. File is required for using vector search engine."
		
		# Verify that the model is supported with config.json.
		with open("config.json", "r") as f:
			config = json.load(f)

		self.config = config
		valid_models = config["models"]
		valid_model_names = list(valid_models.keys())
		assert model in valid_model_names,\
			f"Expected embedding model to be from valid models list {', '.join(valid_model_names)}. Received {model}."
		
		# Verify the model passed in matches the set model in the 
		# config.
		set_model = config["vector-search_config"]["model"]
		assert model == set_model,\
			f"Argument 'model' expected to match 'model' from 'vector-search_config' in config.json. Received {model}."

		# Load model config data.
		self.model_name = model
		self.model_config = valid_models[model]

		# Load model and tokenizer.
		self.device = device
		self.tokenizer, self.model = load_model(config, device=device)

		# Assert that the index directory path string is not empty.
		assert index_dir != "",\
			"Argument 'index_dir' expected to be a valid directory path."
		self.index_dir = index_dir

		# Initialize the vector database.
		self.initialize_vector_db(config)


	def initialize_vector_db(self, config: Dict) -> None:
		# Initialize index directory if it doesn't already exist.
		if not os.path.exists(self.index_dir):
			os.makedirs(self.index_dir, exist_ok=True)

		# Initialize (if need be) and connect to the vector database.
		uri = config["vector-search_config"]["db_uri"]
		self.db = lancedb.connect(uri)

		# Load model dims to pass along to the schema init.
		self.dims = config["models"][self.model_name]["dims"]

		# Initialize schema (this will be passed to the database when 
		# creating a new, empty table in the vector database).
		self.schema = pa.schema([
			pa.field("file", pa.utf8()),
			pa.field("sha1", pa.utf8()),
			pa.field("text_idx", pa.int32()),
			pa.field("text_len", pa.int32()),
			pa.field("vector", pa.list_(pa.float32(), self.dims))
		])


	def search(self, query: str, max_results: int = 50, document_ids: List = [], docs_are_results: bool = False):
		'''
		Conducts a search on the wikipedia data with vector search.
		@param: query str, the raw text that is being queried from the
			wikipedia data.
		@param: max_results (int), the maximum number of search results
			to return.
		@param: document_ids (List), the list of all document (paths)
			that are to be queried from the vector database/indices.
			Can also the the results list from stage 1 search if called
			from ReRank object.
		@param: docs_are_results (bool), a flag as to whether to the
			document_ids list passed in is actually stage 1 search 
			results. Default is False.
		@return: returns a list of objects where each object contains
			an article's path, title, retrieved text, and the slices of
			that text that is being returned (for BM25 and TF-IDF, 
			those slices values are for the whole article).
		'''

		# TODO/NOTE:
		# Current implementation of search involves "dynamic" embedding
		# generation (meaning we generate embeddings at search 
		# runtime). The plan was to originally have preprocess.py 
		# generate the embeddings for the corpus so that lookup was 
		# much faster with lancedb, however, the storage required far
		# exceeded the general storage abilities of most consumer PCs
		# as well as required substantial runtime to generate all 
		# embeddings (if waiting around two weeks for the bag-of-words 
		# preprocessing felt slow, then vector preprocessing was going 
		# to be glacial in comparison). Hence why dynamic embedding 
		# generation is implemented but has issues with scaling with 
		# max_results. For this reason, a hard limit for max_results
		# and the length of the document ids is set.

		assert len(document_ids) > 0,\
			f"Argument 'document_ids' is expected to be not empty. Received {document_ids}"

		# If the hard limit for the number of document ids or 
		# max_results is reached, print an error message and return an
		# empty results list.
		MAX_LIMIT = 10_000
		if len(document_ids) > MAX_LIMIT or max_results > MAX_LIMIT:
			print(f"Number of document_ids or max_results is too high. Hard limit of {MAX_LIMIT} for either.")
			return []
		
		# If the documents passed in are stage 1 search results, copy 
		# the results to their own variable and reset the document ids
		# list to be the document ids in those results.
		if docs_are_results:
			results = copy.deepcopy(document_ids)
			document_ids = [result[1] for result in results]
		
		# Hash the query. This hash will serve as the table name for
		# the database.
		table_name = hashSum(query)
		current_table_names = self.db.table_names()
		assert table_name not in current_table_names,\
			f"Table hash was expected to not exist in database"
		
		# Initialize the fresh table for the current query.
		self.db.create_table(table_name, schema=self.schema)
		table = self.db.open_table(table_name)

		# NOTE:
		# Assumes query text will exist within model tokenizer's max 
		# length. There might be complications for longer queries.
		print("Running Vector search...")

		# Embed the query text.
		query_embedding = self.embed_text(query)

		# Iterate through the document ids.
		for doc_idx in tqdm(range(len(document_ids))):
			document_id = document_ids[doc_idx]
			document, sha1 = os.path.basename(document_id).split(".xml")
			document += ".xml"

			# Load the article text. Loading from stage 1 search 
			# results is faster than loading from file.
			if docs_are_results:
				article_text = results[doc_idx][2]
			else:
				article_text = load_article_text(document, [sha1])[0]

			# Preprocess text (chunk it) for embedding.
			chunk_metadata = vector_preprocessing(
				article_text, self.config, self.tokenizer
			)

			# Embed each chunk and update the metadata.
			for idx, chunk in enumerate(chunk_metadata):
				# Update/add the metadata for the source filename
				# and article SHA1.
				chunk.update({"file": document, "sha1": sha1})

				# Get original text chunk from text.
				text_idx = chunk["text_idx"]
				text_len = chunk["text_len"]
				text_chunk = article_text[text_idx: text_idx + text_len]

				# Embed the text chunk.
				embedding = self.embed_text(text_chunk)

				# NOTE:
				# Originally I had embeddings stored into the metadata
				# dictionary under the "embedding", key but lancddb
				# requires the embedding data be under the "vector"
				# name.

				# Update the chunk dictionary with the embedding
				# and set the value of that chunk in the metadata
				# list to the (updated) chunk.
				# chunk.update({"embedding": embedding})
				chunk.update({"vector": embedding})
				chunk_metadata[idx] = chunk

		# Add chunk metadata to the vector database. Should be on
		# "append" mode by default.
		table.add(chunk_metadata, mode="append")

		# Search the table.
		results = table.search(query_embedding).limit(max_results)
		results = results.to_list()

		# Format search results.
		results = [
			tuple([
				result["_distance"], 
				result["file"] + result["SHA1"], 
				load_article_text(result["file"], [result["SHA1"]]),
				[
					result["text_idx"], 
					result["text_idx"] + result["text_len"]
				]
			])
			for result in results
		]

		# Clear table.
		self.db.drop_table(table_name)

		# Return the search results.
		return results


	def embed_text(self, text: str):
		# Disable gradients.
		with torch.no_grad():
			# Pass original text chunk to tokenizer. Ensure the data is
			# passed to the appropriate (hardware) device.
			output = self.model(
				**self.tokenizer(
					text,
					add_special_tokens=False,
					padding="max_length",
					return_tensors="pt"
				).to(self.device)
			)

			# Compute the embedding by taking the mean of the last 
			# hidden state tensor across the seq_len axis.
			embedding = output[0].mean(dim=1)

			# Apply the following transformations to allow the
			# embedding to be compatible with being stored in the
			# vector DB (lancedb):
			#	1) Send the embedding to CPU (if it's not already
			#		there)
			#	2) Convert the embedding to numpy and flatten the
			# 		embedding to a 1D array
			embedding = embedding.to("cpu")
			embedding = embedding.numpy()[0]
		
		# Return the embedding.
		return embedding


class ReRankSearch:
	def __init__(self, bow_path: str, index_path: str, model: str, 
			  	corpus_size: int=-1, avg_doc_len: float=-1.0,
				srt: float = -1.0, use_json: bool = False, 
				k1: float = 1.0, b: float = 0.0, device: str = "cpu",
				use_tf_idf: bool = False):
		# Set class variables.
		self.bow_dir = bow_path
		self.index_dir = index_path
		self.model = model
		self.corpus_size = corpus_size
		self.avg_corpus_len = avg_doc_len
		self.srt = srt
		self.use_json = use_json
		self.k1 = k1
		self.b = b
		self.use_tfidf = use_tf_idf
		self.device = device

		# Initialize search objects.
		self.tf_idf, self.bm25 = None, None
		if use_tf_idf:
			self.tf_idf = TF_IDF(
				self.bow_dir, self.corpus_size, self.srt, 
				use_json=self.use_json
			)
		else:
			self.bm25 = BM25(
				self.bow_dir, k1=self.k1, b=self.b, 
				corpus_size=self.corpus_size, 
				avg_doc_len=self.avg_corpus_len,
				srt=self.srt, use_json=self.use_json
			)
		self.vector_search = VectorSearch(
			self.model, self.index_dir, self.device
		)

		# Organize search into stages.
		self.stage1 = self.tf_idf if self.use_tfidf else self.bm25
		self.stage2 = self.vector_search


	def search(self, query: str, max_results: int = 50):
		# Pass the search query to the first stage.
		stage_1_results = self.stage1.search(
			query, max_results=max_results
		)

		# Return the first stage search results if the results are empty.
		if len(stage_1_results) == 0:
			return stage_1_results

		# document_ids = [
		# 	# result["document_path"] for result in stage_1_results
		# 	result[1] for result in stage_1_results
		# ]

		# From the first stage, isolate the document paths to target in
		# the vector search.
		stage_2_results = self.stage2.search(
			query, max_results=max_results, document_ids=stage_1_results,
			docs_are_results=True
		)

		# Return the search results from the second stage.
		return stage_2_results