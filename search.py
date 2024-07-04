# search.py
# Implement search methods on the dowloaded (preprocessed) wikipedia
# data.
# Python 3.9
# Windows/MacOS/Linux

import os
import json
import math
from typing import List, Dict

from bs4 import BeautifulSoup
import faiss
import lancedb
import msgpack
import numpy as np

from preprocess import load_model
from preprocess import bow_preprocessing, vector_preprocessing


def load_article_file(path: str):
	pass


def load_data_from_msgpack(path: str):
	with open(path, 'rb') as f:
		byte_data = f.read()

	return msgpack.unpackb(byte_data)


def load_data_from_json(path: str):
	with open(path, "r") as f:
		return json.load(f)
	

def load_data_file(path: str, use_json: bool = False):
	if use_json:
		return load_data_from_json(path)
	return load_data_from_msgpack(path)


def cosine_similarity(vec1: List[float], vec2: List[float]):
	np_vec1 = np.array(vec1)
	np_vec2 = np.array(vec2)
	cosine = np.dot(np_vec1, np_vec2) /\
		(np.linalg.norm(np_vec1) * np.linalg.norm(np_vec2))
	return cosine
	

class BagOfWords: 
	def __init__(self, bow_dir: str, srt: float=-1.0, use_json=False) -> None:
		# Initialize class variables either from arguments or with
		# default values.
		self.bow_dir = bow_dir
		self.word_to_doc_folder = None
		self.doc_to_word_folder = None
		self.word_to_doc_files = None
		self.doc_to_word_files = None
		self.corpus_size = 0	# total number of documents (articles)
		self.use_json = use_json
		self.extension = ".json" if use_json else ".msgpack"
		self.valid_aggr = ["sum", "mean"]

		# Initialize mapping folder path and files list.
		self.locate_and_validate_documents(bow_dir)

		# Verify that the class variables that were initialized as None
		# by default are no longer None.
		initialized_variables = [
			self.word_to_doc_folder, self.doc_to_word_folder,
			self.word_to_doc_files, self.doc_to_word_files
		]
		assert None not in initialized_variables,\
			"Some variables were not initialized properly"
		
		# Initialize the corpus size.
		self.corpus_size = self.get_number_of_documents()

		# Verify that the corpus size is not 0.
		assert self.corpus_size != 0,\
			"Could not count the number of documents (articles) in the corpus. Corpus size is 0."


	def locate_and_validate_documents(self, bow_dir):
		# Initialize path to word to document and document to word 
		# folders.
		self.word_to_doc_folder = os.path.join(bow_dir, 'word_to_docs')
		self.doc_to_word_folder = os.path.join(bow_dir, 'doc_to_words')

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
		
		# Verify that the list of files for each mapping folder is not
		# empty.
		assert len(self.word_to_doc_files) != 0,\
			f"Detected word to documents folder {self.word_to_doc_folder} to have not supported files"
		assert len(self.word_to_doc_files) != 0,\
			f"Detected document to words folder {self.doc_to_word_folder} to have not supported files"
		

	def get_number_of_documents(self):
		counter = 0
		for file in self.doc_to_word_files:
			doc_to_words = load_data_from_msgpack(file)
			counter += len(list(doc_to_words.keys()))
		
		return counter
	

	def compute_tf(self, doc_to_words: Dict, words: List[str]):
		# total_word_count, word_freq
		# doc, word
		doc_tf = dict()
		word_freq_map = dict()
		word_vec = []

		# Iterate through each document.
		for doc in doc_to_words:
			# Initialize the document's word vector.
			word_vec = []

			# Extract the document owrd frequencies.
			word_freq_map = doc_to_words[doc]

			# Compute total word count.
			total_word_count = sum(
				[value for value in word_freq_map.values()]
			)

			# Compute the term frequency accordingly and add it to the 
			# document's word vector
			for word in words:
				if word in word_freq_map:
					word_freq = word_freq_map[word]
					word_vec.append(word_freq / total_word_count)
				else:
					word_vec.append(0)

			doc_tf[doc] = word_vec
			
		# Return the dictionary of the document term frequency word
		# vectors.
		return doc_tf
	

	def compute_idf(self, words: List[str]):
		# Initialize a dictionary containing the mappings of query 
		# words to the total count of how many articles each appears 
		# in.
		word_count = {word: 0 for word in words}

		# Iterate through each file.
		for file in self.word_to_doc_files:
			# Load the word to doc mappings from file.
			word_to_docs = load_data_file(file, use_json=self.use_json)

			# Iterate through each word. Update the total count for
			# each respective word if applicable.
			for word in words:
				if word in word_to_docs:
					word_count[word] += word_to_docs[word]

		# Compute inverse document frequency for each term.
		# return [
		# 	math.log(self.corpus_size / word_count[word])
		# 	for word in words
		# ]
		return {
			word: math.log(self.corpus_size / word_count[word])
			for word in words
		}


class TF_IDF(BagOfWords):
	def __init__(self, bow_dir: str, srt: float=-1.0, use_json=False) -> None:


		self.srt = srt
		self.use_json = use_json
		pass


	def search(self, query:str, max_results:int = 50):
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

		query_tfidf, corpus_tfidf = self.compute_tfidf(
			words, word_freq
		)

		# Convert the dictionary to a sorted list (sorted from largest
		# to smallest TF-IDF vector sum).
		# sorted_list = sorted(
		# 	docs_to_tfidf.items(), 
		# 	key=lambda item: item[1], 
		# 	reverse=True
		# )

		# Trim the list by the max_results value.
		# sorted_list = sorted_list[:max_results]

		# Return the list.
		# return sorted_list
	

	def compute_tfidf(self, words: List[str], query_word_freq: Dict):
		# Sort the set of words (ensures consistent positions of each 
		# word in vector).
		words = sorted(words)

		# Iterate through all files to get IDF. Compute only once, not 
		# per file.
		word_idf = self.compute_idf(words)

		# Compute query TF-IDF.
		query_tf = dict()
		query_tfidf = dict()
		query_total_word_count = sum([value for value in query_word_freq.values()])
		for word in words:
			query_tf[word] = query_word_freq[word] / query_total_word_count
			query_tfidf[word] = query_tf[word] * word_idf[word]

		# Compute corpus TF-IDF.
		tf_idf = dict()

		# Compute TF-IDF for every article.
		for file in self.doc_to_word_files:
			# Load the doc to word frequency mappings from file.
			doc_to_words = load_data_file(file, self.use_json)

			# Compute article TF.
			# doc_tf = self.compute_tf(doc_to_words, words)

			# Compute TF-IDF.

			# Iterate through each document.
			for doc in doc_to_words:
				# Initialize the document's word vector.
				# word_vec = []
				word_vec = dict()

				# Extract the document owrd frequencies.
				word_freq_map = doc_to_words[doc]

				# Compute total word count.
				total_word_count = sum(
					[value for value in word_freq_map.values()]
				)

				# Compute the term frequency accordingly and add it to the 
				# document's word vector
				for word in words:
					if word in word_freq_map:
						word_freq = word_freq_map[word]
						# word_vec.append(word_freq / total_word_count)
						word_vec[word] = word_freq / total_word_count
					else:
						# word_vec.append(0)
						word_vec[word] = 0

				# doc_tf[doc] = word_vec
				tf_idf[doc] = {
					word: word_vec[word] * word_idf[word] 
					for word in words
				}

		print("query TF-IDF:")
		print(json.dumps(query_tfidf, indent=4))
		print("top 5 document TF-IDF:")
		print(
			json.dumps(
				{
					doc: value for doc, value in tf_idf.items()
					if doc in list(tf_idf.keys())[:5]
				}, 
				indent=4
			)
		)


class BM25(BagOfWords):
	def __init__(self, bow_dir: str, srt: float=-1.0) -> None:

		pass


	def search(self, query, max_results=50):
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
		pass


class VectorSearch:
	def __init__(self, model="bert-base-uncased", index_dir="./vector_indices"):
		# Dictionary of supported models for generating vector
		# embeddings. Each model is going to be supplied with its
		# desired local storage path (for both model and tokenizer) and
		# the maximum context length (in tokens) that it supports.
		self.supported_models = {
			"bert-base-uncased": {
				"path": "./BERT",
				"context_length": 512,
			},
		}

		# Assert that the model passed in is one of the supported ones.
		supported_model_names = list(self.supported_models.keys())
		assert model in supported_model_names, f"Embedding model {model} is not in list of supported models {supported_model_names}"

		# Assert that the index directory exists.
		assert os.path.exists(index_dir) and os.path.isdir(index_dir), f"Index directory {index_dir} does not exist."

		# Assert that the index directory is populated with (faiss) 
		# index files.
		self.index_files = [
			os.path.join(index_dir, file) 
			for file in os.listdir(index_dir) 
			if file.endswith("pkl")
		]
		assert len(self.index_files) != 0, f"No index files found in {index_dir}"

		# Assert that the index directory contains a .
		map_file = "document_map.json"
		self.index_map_path = os.path.join(index_dir, map_file)
		assert os.path.exists(self.index_map_path) and os.path.isfile(self.index_map_path), f"No document map file ({map_file}) was found in {index_dir}"

		# Load the model. 
  
		# Load the index map file.
		with open(self.index_map_path, "r") as imp_f:
			self.index_map = json.load(imp_f)


	def search(self, query, max_results=50, document_ids=[]):
		'''
		Conducts a search on the wikipedia data with vector search.
		@param: query str, the raw text that is being queried from the
			wikipedia data.
		@param: max_results (int), the maximum number of search results
			to return.
		@param: document_ids (List[str]), the list of all document 
			(paths) that are to be queried from the vector 
			database/indices.
		@return: returns a list of objects where each object contains
			an article's path, title, retrieved text, and the slices of
			that text that is being returned (for BM25 and TF-IDF, 
			those slices values are for the whole article).
		'''

		# Perform a global search if document_ids is an empty list. 
		# Otherwise, perform a targeted search across the index, 
		# creating a new, temporary index composed of only documents 
		# from the document_ids list.
		index_list = self.index_files
		if len(document_ids) != 0:
			pass

		pass


class ReRankSearch:
	def __init__(self, bow_path, index_path, model, max_results=50, srt=None, use_tf_idf=False):
		# Set class variables.
		self.bow_dir = bow_path
		self.index_dir = index_path
		self.model = model
		self.max_results = max_results
		self.srt = srt
		self.use_tfidf = use_tf_idf

		# Initialize search objects.
		self.bm25 = BM25(self.bow_dir, self.srt)
		self.tf_idf = TF_IDF(self.bow_dir, self.srt)
		self.vector_search = VectorSearch(self.model, self.index_dir)

		# Organize search into stages.
		self.stage1 = self.bm25 if not use_tf_idf else self.tf_idf
		self.stage2 = self.vector_search


	def search(self, query, max_results=50):
		# Pass the search query to the first stage.
		stage_1_results = self.stage1.search(
			query, max_results=max_results
		)

		# Return the first stage search results if the results are empty.
		if len(stage_1_results) == 0:
			return stage_1_results

		document_ids = [
			result["document_path"] for result in stage_1_results
		]

		# From the first stage, isolate the document paths to target in
		# the vector search.
		stage_2_results = self.stage2.search(
			query, max_results=max_results, document_ids=document_ids
		)

		# Return the search results from the second stage.
		return stage_2_results