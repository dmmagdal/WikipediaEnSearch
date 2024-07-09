# search.py
# Implement search methods on the dowloaded (preprocessed) wikipedia
# data.
# Python 3.9
# Windows/MacOS/Linux


import heapq
import json
import math
import os
from typing import List, Dict, Tuple

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import faiss
import lancedb
import msgpack
# from nltk.tokenize import word_tokenize
import numpy as np

from preprocess import load_model, process_page
from preprocess import bow_preprocessing, vector_preprocessing


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

	# Initiaize the return list of articles.
	articles = []

	# Iterate through the different SHA1 values from the SHA1 hash
	# list.
	for sha1_hash in sha1_list:
		# Isolate the target article with the given SHA1 hash. Append
		# the processed text if it was found, otherwise append the
		# error message string.
		page = soup.find("page", attrs={"sha1": sha1_hash})
		# page = soup.find("page", sha1=sha1_hash)
		if page is not None:
			articles.append(process_page(page))
		else:
			articles.append(f"ERROR: COULD NOT LOCATE ARTICLE {sha1_hash} in {path}")

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
	

def print_results(results: List, search_type:str = "tf-idf") -> None:
	pass


class BagOfWords: 
	def __init__(self, bow_dir: str, srt: float=-1.0, use_json=False) -> None:
		'''
		Initialize a Bag-of-Words search object.
		@param: bow_dir (str), a path to the directory containing the 
			bag of words metadata. This metadata includes the folders
			mapping to the word-to-document and document-to-word files.
		@param: srt (float), the similarity relative threshold. A value
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
		self.corpus_size = 0	# total number of documents (articles)
		self.srt = srt			# similarity relative threshold value
		self.use_json = use_json
		self.extension = ".json" if use_json else ".msgpack"

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
		for file in self.doc_to_word_files:
			# Load the data from the file and increment the counter by
			# the number of documents in each file.
			doc_to_words = load_data_file(file, self.use_json)
			counter += len(list(doc_to_words.keys()))
		
		# Return the count.
		return counter
	

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
		Compute the Inverse Document Frquency of the given set of 
			(usually query) words.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: returns the Inverse Document Frequency for all words
			queried in the corpus. The data is returned in an ordered
			list (List[float]) where the index of each value
			corresponds to the index of a word in the word list 
			argument.
		'''
		# Initialize a list containing the mappings of the query words
		# to the total count of how many articles each appears in.
		word_count = [0.0] * len(words)

		# Iterate through each file.
		for file in self.word_to_doc_files:
			# Load the word to doc mappings from file.
			word_to_docs = load_data_file(file, use_json=self.use_json)

			# Iterate through each word. Update the total count for
			# each respective word if applicable.
			for word_idx in range(len(words)):
				word = words[word_idx]
				if words[word] in word_to_docs:
					word_count[word_idx] += word_to_docs[word]

		# Compute inverse document frequency for each term.
		return [
			math.log(self.corpus_size / word_count[word_idx])
			if word_count[word_idx] != 0.0 else 0.0
			for word_idx in range(len(words))
		]


class TF_IDF(BagOfWords):
	def __init__(self, bow_dir: str, srt: float=-1.0, use_json=False) -> None:
		super().__init__(bow_dir=bow_dir, srt=srt, use_json=use_json)
		# self.bow_dir = bow_dir
		# self.srt = srt
		# self.use_json = use_json
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

		# Compute the TF-IDF for the corpus.
		_, corpus_tfidf = self.compute_tfidf(
			words, word_freq, max_results=max_results
		)

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
			document += ".xml"
			text = load_article_text(document, [sha1])
			
			# Append the results,
			full_result = result + tuple([text, [0, len(text)]])

			# Insert the item into the list from the front.
			sorted_rankings.insert(0, full_result)

		# Return the list.
		return sorted_rankings
	

	def compute_tfidf(self, words: List[str], query_word_freq: Dict, max_results: int = -1.0):
		'''
		Iterate through all the documents in the corpus and compute the
			TF-IDF for each document in the corpus. Sort the results
			based on the cosine similarity score and return the sorted
			list.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: query_word_freq (Dict), the word frequency mapping for 
			the input search query.
		@param: max_results (int), the maximum number of results to 
			return. Default is -1.0 (no limit).
		@return: returns the TF-IDF for the query as well as the sorted
			list of search results. 
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

		# NOTE:
		# Heapq in use is a max-heap. This is implemented by 
		# multiplying the cosine similarity score by -1. That way, the
		# largest values are actually the smallest in the heap and are
		# popped when we need to pushpop the largest scoring tuple.

		# Compute TF-IDF for every file.
		for file in self.doc_to_word_files:
			# Load the doc to word frequency mappings from file.
			doc_to_words = load_data_file(file, self.use_json)

			# Iterate through each document.
			for doc in doc_to_words:
				# Extract the document owrd frequencies.
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

				# If the similarity relevence threshold has been 
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
				if max_results != -1.0 and len(corpus_tfidf_heap) >= max_results:
					# Pushpop the highest (cosine similarity) value
					# tuple from the heap to make way for the next
					# tuple.
					heapq.heappushpop(
						corpus_tfidf_heap,
						tuple([doc_cos_score, doc, doc_tfidf])
					)
				else:
					heapq.heappush(
						corpus_tfidf_heap,
						tuple([doc_cos_score, doc, doc_tfidf])
					)

		# Return the query TF-IDF and the corpus TF-IDF.
		return query_tfidf, corpus_tfidf_heap


class BM25(BagOfWords):
	def __init__(self, bow_dir: str, k1: float = 1.5, b: float = 0.75, 
			  	srt: float=-1.0, use_json=False) -> None:
		super().__init__(bow_dir=bow_dir, srt=srt, use_json=use_json)
		self.avg_corpus_len = self.compute_avg_corpus_size()
		self.k1 = k1
		self.b = b
		pass


	def compute_avg_corpus_size(self) -> float:
		# Initialize document size sum.
		doc_size_sum = 0

		# Iterate through each file in the documents to words map 
		# files.
		for file in self.doc_to_word_files:
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
		words, word_freq = bow_preprocessing(query, True)

		# Compute the BM25 for the corpus.
		corpus_bm25 = self.compute_bm25(words, word_freq, max_results=max_results)

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
			document += ".xml"
			text = load_article_text(document, [sha1])
			
			# Append the results,
			full_result = result + tuple([text, [0, len(text)]])

			# Insert the item into the list from the end (append).
			sorted_rankings.append(full_result)
		
		# Return the list.
		return sorted_rankings


	def compute_bm25(self, words: List[str], query_word_freq: Dict, max_results: int = -1.0):
		'''
		Iterate through all the documents in the corpus and compute the
			BM25 for each document in the corpus. Sort the results
			based on the cosine similarity score and return the sorted
			list.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: query_word_freq (Dict), the word frequency mapping for 
			the input search query.
		@param: max_results (int), the maximum number of results to 
			return. Default is -1.0 (no limit).
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

		# NOTE:
		# Heapq in use is a max-heap. In this case, we don't want to 
		# multiply the BM25 score by -1 because a larger score means a
		# document is "more relevant" to the query (so we want to drop
		# the lower scores if we have a max_results limit). BM25 also 
		# doesn't require using cosine similarity since it aggregates
		# the term values into a sum for the document score.

		# Compute BM25 for every file.
		for file in self.doc_to_word_files:
			# Load the doc to word frequency mappings from file.
			doc_to_words = load_data_file(file, self.use_json)

			# Iterate through each document.
			for doc in doc_to_words:
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

				# Insert the document name (includes file path & 
				# article SHA1), BM25 score to the heapq. The heapq 
				# sorts by the first value in the tuple so that is why
				# the cosine similarity score is the first item in the 
				# tuple.
				if max_results != -1.0 and len(corpus_bm25_heap) >= max_results:
					# Pushpop the smallest (BM25) value tuple from the 
					# heap to make way for the next tuple.
					heapq.heappushpop(
						corpus_bm25_heap,
						tuple([bm25_score, doc])
					)
				else:
					heapq.heappush(
						corpus_bm25_heap,
						tuple([bm25_score, doc])
					)

		# Return the corpus BM25 rankings.
		return corpus_bm25_heap


class VectorSearch:
	def __init__(self, model="bert-base-uncased", index_dir="./vector_indices", in_memory=True):
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


	def search(self, query: str, max_results: int = 50, document_ids: List[str] = []):
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