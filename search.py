# search.py
# Implement search methods on the dowloaded (preprocessed) wikipedia
# data.
# Python 3.9
# Windows/MacOS/Linux

import os
import json
import math
from bs4 import BeautifulSoup
import faiss
from transformers import AutoTokenizer
from preprocess import bow_preprocessing


def load_article_file(path: str):
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


class BM25:
	def __init__(self, bow_dir: str, srt: float=-1.0) -> None:

		doc_to_word = dict()
		word_to_doc = dict()

		# Compute corpus size (number of documents) and average
		# document length.
		self.docs = list(doc_to_word.keys())
		self.num_docs = len(self.docs)
		self.avg_doc_len = sum([
			sum(doc_to_word[doc].values()) for doc in self.docs
		]) / self.num_docs
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


class TF_IDF:
	def __init__(self, bow_dir: str, srt: float=-1.0) -> None:
		# Load the compiled mappings.
		w2d_folder_path = os.path.join(bow_dir, "word_to_docs")
		d2w_folder_path = os.path.join(bow_dir, "doc_to_words")

		w2d_files = [
			file for file in os.listdir(w2d_folder_path)
			if file.endswith(".json")
		]
		d2w_files = [
			file for file in os.listdir(d2w_folder_path)
			if file.endswith(".json")
		]

		self.words_to_docs = dict()
		self.docs_to_words = dict()

		for w2d_file in w2d_files:
			with open(w2d_file, "r") as f1:
				self.words_to_docs.update(json.load(f1))
		
		for d2w_file in d2w_files:
			with open(d2w_file, "r") as f2:
				self.words_to_docs.update(json.load(f2))

		self.srt = srt
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
		words = bow_preprocessing(query)

		# Isolate the list of documents and the total number of 
		# documents.
		docs = list(self.docs_to_words)
		num_docs = len(docs)

		# Initialize the mapping from documents to their TF-IDF values.
		docs_to_tfidf = dict()

		# Iterate throuch each document, computing the TF-IDF for each.
		for doc in docs:
			# Initialize the vector sum for all words in the search.
			vector_sum = 0.0

			# Iterate through every word in the search query.
			for word in words:
				# Compute text frequency.
				word_freq_in_doc = self.docs_to_words[doc][word]
				total_word_in_doc = sum(
					self.docs_to_words[doc].values()
				)	# total words in document = sum of all word frequencies in the document
				text_freq = word_freq_in_doc / total_word_in_doc

				# Compute inverse document frequency.
				num_docs_with_word = len(self.words_to_docs[word])
				inverse_doc_freq = math.log(
					num_docs / num_docs_with_word
				)

				# Put together TF-IDF.
				tf_idf = text_freq * inverse_doc_freq
				vector_sum += tf_idf

			# Map the total word vector sum to the document.
			docs_to_tfidf[doc] = vector_sum

		# Convert the dictionary to a sorted list (sorted from largest
		# to smallest TF-IDF vector sum).
		sorted_list = sorted(
			docs_to_tfidf.items(), 
			key=lambda item: item[1], 
			reverse=True
		)

		# Trim the list by the max_results value.
		sorted_list = sorted_list[:max_results]

		# Return the list.
		return sorted_list


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