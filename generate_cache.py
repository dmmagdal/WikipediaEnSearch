# generate_cache.py
# Compute the word level term frequency as well as TF-IDF for each 
# document in the corpus and save it to a cache so that search.py only
# has to read it and minimizes the compute. This is another 
# preprocessing script that should be run after preprocess.py but 
# before searchwiki.py.


import argparse
from argparse import Namespace
import gc
import json
import math
import multiprocessing as mp
import os
from typing import List, Dict

import msgpack
from tqdm import tqdm



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


def get_number_of_documents(doc_to_word_files: List[str], use_json: bool = False) -> int:
		'''
		Count the number of documents recorded in the corpus.
		@param, .
		@param, .
		@return, Returns the number of documents in the corpus.
		'''
		# Initialize the counter to 0.
		counter = 0

		# Iterate through each file in the documents to words map 
		# files.
		for file in tqdm(doc_to_word_files):
			# Load the data from the file and increment the counter by
			# the number of documents in each file.
			doc_to_words = load_data_file(file, use_json)
			counter += len(list(doc_to_words.keys()))
		
		# Return the count.
		return counter


def compute_idf(word_to_doc_files: Dict, corpus_size: int, words: List[str], use_json: bool = False) -> Dict:
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
	# word_count = [0.0] * len(words)
	word_count = {word: 0.0 for word in words}

	# Iterate through each file.
	for file in tqdm(word_to_doc_files):
		# Load the word to doc mappings from file.
		word_to_docs = load_data_file(file, use_json)

		# Iterate through each word. Update the total count for
		# each respective word if applicable.
		# for word_idx in range(len(words)):
		# 	word = words[word_idx]
		# 	if word in word_to_docs:
		# 		word_count[word_idx] += word_to_docs[word]
		for word in words:
			if word in word_to_docs:
				word_count[word] += word_to_docs[word]

	# Compute inverse document frequency for each term.
	# return [
	# 	math.log(corpus_size / word_count[word_idx])
	# 	if word_count[word_idx] != 0.0 else 0.0
	# 	for word_idx in range(len(words))
	# ]
	return {
		word: math.log(corpus_size / word_count[word])
		if word_count[word] != 0.0 else 0.0
		for word in words
	}


def merge_results(results):
	# Initialize aggregate variable.
	aggr_doc_to_word = dict()

	# Results mappings shape (num_processors, tuple_len). Iterate
	# through each result and update the aggregate variables.
	for result in results:
		# Unpack the result tuple.
		doc_to_word = result

		# Update the document to word dictionary. Just call a
		# dictionary's update() function here since every key in the
		# entirety of the results is unique.
		aggr_doc_to_word.update(doc_to_word)

	# Return the aggregated data.
	return aggr_doc_to_word


def multiprocess_metadata(words_idf: Dict, doc_to_words: Dict, num_proc: int = 1):
	# Break down the document to word mappings into chunks.
	documents = list(doc_to_words.keys())
	chunk_size = math.ceil(len(documents) / num_proc)
	chunks = [
		{
			doc: doc_to_words[doc] 
			for doc in documents[i:i + chunk_size]
		}
		for i in range(0, len(documents), chunk_size)
	]

	# Define the arguments list.
	arg_list = [(words_idf, chunk) for chunk in chunks]

	# Distribute the arguments among the pool of processes.
	with mp.Pool(processes=num_proc) as pool:
		# Aggregate the results of processes.
		results = pool.starmap(process_metadata, arg_list)

		# Pass the aggregate results tuple to be merged.
		file_idf= merge_results(results)

	# Return the different mappings.
	return file_idf


def process_metadata(words_idf: Dict, doc_to_words: Dict):
	file_tfidf = dict()

	# Iterate through all documents in the doc-to-words mapping.
	for document in tqdm(list(doc_to_words.keys())):
		document_tfidf = dict()

		# Extract the document word frequencies.
		word_freq_map = doc_to_words[document]

		# Compute the document size (in words).
		doc_len = sum([value for value in word_freq_map.values()])

		# Iterate through the list of words.
		for word in list(words_idf.keys()):
			# Skip words that do not appear in the document (TF-IDF is 
			# 0).
			if word not in list(word_freq_map.keys()):
				continue

			# Compute the term frequency.
			word_tf = word_freq_map[word] / doc_len

			# Compute the TF-IDF.
			word_tfidf = word_tf * words_idf[word]

			# Store values to document dictionary.
			document_tfidf[word] = {
				"TF": word_tf,
				"TF-IDF": word_tfidf
			}

		# Store values to document.
		file_tfidf[document] = {
			"word_TF-IDF": document_tfidf,
			"document_length": doc_len
		}

	# Return the mappings for the file.
	return file_tfidf


def main():
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
		'--num_proc', 
		type=int, 
		default=1, 
		help="Number of processor cores to use for multiprocessing. Default is 1."
	)
	parser.add_argument(
		"--use_json",
		action="store_true",
		help="Specify whether to load and write metadata to/from JSON files. Default is false/not specified."
	)
	args = parser.parse_args()

	###################################################################
	# VERIFY METADATA FILES
	###################################################################
	# Open config file.
	with open("config.json", "r") as f:
		config = json.load(f)

	extension = ".json" if args.use_json else ".msgpack"

	# Pull directory paths from the config file.
	preprocessing = config["preprocessing"]
	d2w_metadata_path = preprocessing["doc_to_words_path"]
	w2d_metadata_path = preprocessing["word_to_docs_path"]
	cache_metadata_path = preprocessing["tf_idf_cache_path"]

	# Verify metadata directory paths exist.
	if not os.path.exists(d2w_metadata_path):
		print(f"Bag-of-words document-to-words metadata folder not initialized.")
		print(f"Please initialized folder by downloading the metadata from huggingface.")
		exit(1)

	if not os.path.exists(w2d_metadata_path):
		print(f"Bag-of-words word-to-documents metadata folder not initialized.")
		print(f"Please initialized folder by downloading the metadata from huggingface.")
		exit(1)
	
	# Initialize the directory for the TF-IDF cache if it doesn't 
	# already exist.
	if not os.path.exists(cache_metadata_path):
		os.makedirs(cache_metadata_path, exist_ok=True)

	# NOTE:
	# I tried to make this cleaner but python would throw an error on
	# on the os.listdir() line for the metadata directories if they
	# did not exist. Therefore, it made it impossible to define 
	# w2d_data_files and d2w_data_files before checking for the 
	# existance of the required metadata directories.
	d2w_data_files = sorted(
		[
			os.path.join(d2w_metadata_path, file) 
			for file in os.listdir(d2w_metadata_path)
			if file.endswith(extension)
		]
	)
	w2d_data_files = sorted(
		[
			os.path.join(w2d_metadata_path, file) 
			for file in os.listdir(w2d_metadata_path)
			if file.endswith(extension)
		]
	)

	if len(d2w_data_files) == 0:
		print(f"Bag-of-words document-to-words metadata folder has no files.")
		print(f"Follow the README.md for instructions on how to download the files from huggingface.")
		exit(1)

	if len(w2d_data_files) == 0:
		print(f"Bag-of-words word-to-documents metadata folder has no files.")
		print(f"Follow the README.md for instructions on how to download the files from huggingface.")
		exit(1)

	if len(d2w_data_files) != len(w2d_data_files):
		print(f"Expected the word-to-documents and document-to-words files to match 1-to-1.")
		exit(1)

	d2w_files = sorted(
		[
			file for file in os.listdir(d2w_metadata_path) 
			if file.endswith(extension)
		]
	)
	w2d_files = sorted(
		[
			file for file in os.listdir(w2d_metadata_path) 
			if file.endswith(extension)
		]
	)
	for file_idx in range(len(d2w_data_files)):
		if d2w_files[file_idx] != w2d_files[file_idx]:
			print(f"Expected the word-to-documents and document-to-words files to match 1-to-1.")
			print(f"Make sure all metadata documents have corresponding matches.")
			exit()
	# zipped_list = list(zip(w2d_data_files, d2w_data_files))

	###################################################################
	# PROGRESS CHECK
	###################################################################
	# Progress files.
	progress_file = "./generate_cache_state.txt"

	# Progress list.
	progress = []

	# TODO:
	# Refactor this code to not use the same line(s) twice to 
	# initialize/clear the progress files.

	if args.restart:
		# Clear the progress files if the restart flag has been thrown.
		open(progress_file, "w+").close()
	else:
		# Override progress list with file contents (if the restart
		# flag has not been thrown).
		if os.path.exists(progress_file):
			with open(progress_file, "r") as pf1: 
				progress = pf1.readlines()
			progress = [file.rstrip("\n") for file in progress]
		else:
			open(progress_file, "w+").close()

	# Load corpus size values from config.
	tfidf_corpus_size = config["tf-idf_config"]["corpus_size"]
	bm_corpus_size = config["bm25_config"]["corpus_size"]

	# Compute the corpus size if the values either mismatch or at least
	# one of the values are zero.
	condition1 = tfidf_corpus_size != bm_corpus_size
	condition2 = (tfidf_corpus_size == 0) or (bm_corpus_size == 0)
	if condition1 or condition2:
		# Compute corpus size (number of documents/articles).
		corpus_size = get_number_of_documents(
			d2w_data_files, args.use_json
		)

		# Update config parameter and save.
		config["tf-idf_config"]["corpus_size"] = corpus_size
		config["bm25_config"]["corpus_size"] = corpus_size
		with open("config.json", "w") as f:
			json.dump(config, f, indent=4)

	# Iterate through each file and preprocess it.
	for idx in range(len(d2w_metadata_path)):
		# Isolate the base files.
		base_file = d2w_files[idx]
		d2w_file = d2w_data_files[idx]
		w2d_file = w2d_data_files[idx]

		# Check if the file has already been processed. If so, skip the
		# file.
		if base_file in progress:
			print(f"Already processed file ({idx + 1}/{len(d2w_files)}) {base_file}.")
			continue

		# Open the current document-to-words file.
		doc_to_words = load_data_file(
			d2w_data_files[file_idx], args.use_json
		)

		# Isolate the set of all unique words in the current 
		# document-to-words file.
		words = set()
		for doc in list(doc_to_words.keys()):
			word_freq = doc_to_words[doc]
			doc_words = [word for word in list(word_freq.keys())]
			words.update(doc_words)

		# Convert words set to a list.
		words = sorted(list(words))

		# Compute the IDF for all words.
		word_idfs = compute_idf(w2d_data_files, corpus_size, words, args.use_json)

		if args.num_proc > 1:
			# Determine the number of CPU cores to use (this will be
			# passed down the the multiprocessing function)
			max_proc = min(mp.cpu_count(), args.num_proc)
			tf_idf_data = multiprocess_metadata(
				word_idfs, doc_to_words, num_proc=max_proc
			)
		else:
			tf_idf_data = process_metadata(word_idfs, doc_to_words)

		pass

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()