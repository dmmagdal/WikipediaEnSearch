# semantic_category_inverted_index.py
# Convert all categories from the document dump into semantic vectors
# and use those vectors as part of an inverted index.
# Python 3.11
# Windows/MacOS/Linux


import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import gc
import math
import multiprocessing as mp
import os
import pyarrow as pa
import random
import re
import time
from typing import Dict, List, Any

import lancedb
from lancedb.table import LanceTable
import msgpack
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import psutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from rake_nltk import Rake
from yake import KeywordExtractor
import spacy
import pytextrank

from preprocess import load_model
from preprocess import lowercase, handle_special_numbers, remove_punctuation
from preprocess import remove_stopwords, replace_subscripts, replace_superscripts


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


def write_data_to_msgpack(path: str, data: Dict) -> None:
	'''
	Write data (dictionary) to a msgpack file given the path.
	@param: path (str), the path of the data file that is to be 
		written/created.
	@param: data (Dict), the structured data (dictionary) that is to be
		written to the file.
	@return: returns nothing.
	'''
	with open(path, 'wb+') as f:
		packed = msgpack.packb(data)
		f.write(packed)


def write_data_to_json(path: str, data: Dict) -> None:
	'''
	Write data (dictionary) to a json file given the path.
	@param: path (str), the path of the data file that is to be 
		written/created.
	@param: data (Dict), the structured data (dictionary) that is to be
		written to the file.
	@return: returns nothing.
	'''
	with open(path, "w+") as f:
		json.dump(data, f, indent=4)


def write_data_file(path: str, data: Dict, use_json: bool = False) -> None:
	'''
	Write data (dictionary) to either a JSON or msgpack file given the
		path.
	@param: path (str), the path of the data file that is to be loaded.
	@param: data (Dict), the structured data (dictionary) that is to be
		written to the file.
	@param: use_json (bool), whether to write the data file to a JSON 
		or msgpack (default is False).
	@return: returns nothing.
	'''
	if use_json:
		write_data_to_json(path, data)
	else:
		write_data_to_msgpack(path, data)


def preprocess_category_text(category: str) -> str:
	'''
	Clean the category text before it is stored.
	@param: category (str), the category text string.
	@return: returns the category text string (cleaned).
	'''
	# Remove any "Catagory:" from the beginning of the category string.
	category_substring = "Category:"
	if category.startswith(category_substring):
		category = category[len(category_substring):]

	# Remove period characters ("."). This may interfere with lanceDB
	# SQL filtering.
	# category = category.replace(".", " ")

	# Remove quote characters (double """ and single '') from the 
	# category string (this will interfere with the SQL commands to 
	# query a category).
	category = category.replace("'", "").replace('"', "")

	# Return the cleaned category string.
	return category


def embed_text(tokenizer: AutoTokenizer, model: AutoModel, device: str, text: str) -> np.ndarray:
	'''
	Embed the text with the given tokenizer and model.
	@param: tokenizer (AutoTokenizer), pretrained tokenizer for the 
		model.
	@param: model (AutoModel), pretrained transformers model that 
		generates the embeddings.
	@param: device (str), what device to send the tokenizer and model 
		to.
	@param: text (str), the text to be embedded.
	@return: returns a np.float32 array containing the embedded text.
	'''
	# Pass text to the tokenizer (pad to max length) and give the 
	# tokenized output to the model.
	with torch.no_grad():
		output = model(
			**tokenizer(
				text,
				add_special_tokens=False,
				padding="max_length",
				truncation=True,
				return_tensors="pt"
			).to(device)
		)

		# Post process the embedding (take mean against middle dim, 
		# send to CPU, isolate remaining vector)
		embedding = output[0].mean(dim=1)
		embedding = embedding.to("cpu")
		# embedding = embedding.numpy()[0] # batch size 1
		embedding = embedding.numpy() # batch size n

	# Return embedding.
	return embedding


def embed_all_unseen_categories(
	tokenizer: AutoTokenizer, 
	model: AutoModel, 
	device: str, 
	table: LanceTable, 
	categories: List[str], 
	batch_size: int = 1, 
	check_for_duplicates: bool = True
) -> List[Dict[str, Any]]:
	'''
	Iterate through the list of categories and embed all (unseen) 
		categories so they can be stored in the vector DB table.
	@param: tokenizer (AutoTokenizer), pretrained tokenizer for the 
		model.
	@param: model (AutoModel), pretrained transformers model that 
		generates the embeddings.
	@param: device (str), what device to send the tokenizer and model 
		to.
	@param: table (LanceTable), the vector DB table that is being 
		queried.
	@param: categories (List[str]), the list of categories to check
		against the entries in the vector DB table.
	@param: batch_size (int), the size of the batches of text that will 
		be passed into the embedding model. Default is 1
	@param: check_for_duplicate (bool), whether to check for categories'
		that already exist within the embedding. Default is True.
	@return: returns a List[Dict[str, Any]] containing the unique 
		(unseen) category, embedding pairs.
	'''
	# Lists for tracking batches and returned category, embedding 
	# pairs.
	metadata = []
	batch = []

	# NOTE:
	# pyarrow has a limit of 2GB for every object it is storing. Each
	# entry is comprised of a category, embedding pair. The expected
	# storage of each is computed below.
	# - Category
	#   utf-8 (4 bytes) x 512 characters = 2 kb
	# - Embedding
	#   float32 (4 byte) x 1024 floats = 4 kb
	# The 512 characters comes from the max context length of models
	# currently catalogued in config.json at the time of this writing.
	# the 1024 floats come from the max embedding dimensions of models
	# also catalogued in config.json.
	# This means each entry is 6 kb large. Some additional math down
	# below will show this data will scale up in different chunks.
	# - 1,000 -> 6 mb
	# - 100,000 -> 600 mb
	# - 250,000 -> 1.5 gb
	# - 500,000 -> 3 gb (out of range)
	# - 1,000,000 -> 6 gb (out of range)
	# It stands to reason that the chunk size before saving should be 
	# at the 250,000 or 100,000 marks.

	# Variables to tracking number of batches and aggregated batch 
	# data.
	local_data = []
	local_counter = 0
	append_to_chunk = False

	# NOTE:
	# Originally, this function was going to store all the category, 
	# embedding pairs to a list and return that list, at which point it
	# would be merged with other output lists from any other 
	# processes/threads. However, there seems to be an issue with the
	# size of the array passed over to pyarrow when inserting into the 
	# table (pyarrow apparently has a limit of objects with 2GB, much 
	# less than the memory size of the lists being passed in). To 
	# handle this, rather than return the whole list, store only a
	# chunk at a time and write to the table directly here. This will
	# reduce both memory overhead AND limit the size of the object 
	# passed to pyarrow. Letting the process get interrupted will incur
	# a steep penalty in terms of time because of the removal of 
	# categories already stored in the table (it is recommended to do 
	# this all in one successful pass).

	for node in tqdm(categories):
		if check_for_duplicates:
			# Query the vector DB for the category.
			results = table.search()\
				.where(f"category = '{node}'")\
				.limit(1)\
				.to_list()

			# If the category does exist already, skip the entry.
			if len(results) != 0:
				continue

		# Append the (valid) node to the batch.
		batch.append(node)

		# Embed the category and add it to the vector metadata.
		# embedding = embed_text(tokenizer, model, device, node)
		# metadata.append({"category": node, "vector": embedding})

		# Once the batch has reached the specified batch size (or the 
		# current nodes is the end of the list), begin the batch 
		# embedding process to generate the category, embedding pairs.
		if len(batch) == batch_size or categories.index(node) == len(categories) - 1:
			# Embed the category and add it to the vector metadata.
			embedding = embed_text(tokenizer, model, device, batch)
			for i, category in enumerate(batch):
				# NOTE:
				# A blank string or string with whitespace will result 
				# in vector embeddings with NaN values. This will throw
				# an error. The following statement will check for NaN
				# values in the embeddings and skip those strings.

				# Skip nan embeddings for now.
				if np.any(np.isnan(embedding[i])):
					print(f"Catgory: {category}\nEmbedding: {embedding[i]}")
					print(f"Len of category string: {len(category)}")
					continue

				# Append to one list depending on the number of 
				# categories passed into the function.
				if append_to_chunk:
					local_data.append(
						{"category": category, "vector": embedding[i]}
					)
				else:
					metadata.append(
						{"category": category, "vector": embedding[i]}
					)

			# Reset batch.
			batch = []

			# Increment local counter for number batches.
			local_counter += 1

	# Return the list of all category, embedding pairs computed (if the
	# number of categories embedded did not exceed the set threshold).
	return metadata


def search_table_for_categories(table: LanceTable, categories: List[str], chunk_size: int = 100) -> List[str]:
	'''
	Search the categories in the vector DB table and isolate which 
		categories in the arguments list are already found in the 
		table.
	@param: table (LanceTable), the vector DB table that is being 
		queried.
	@param: categories (List[str]), the list of categories to check
		against the entries in the vector DB table.
	@param: chunk_size (int), the size of the chunks the categories
		list will be broken up into in order to allow for faster 
		querying. Default is 100.
	@return: returns a List[str] containing all categories that were
		found in the vector DB table.
	'''
	metadata = []
	assert chunk_size > 0

	# Chunk the categories list.
	category_chunks = [
		categories[i:i + chunk_size]
		for i in tqdm(range(0, len(categories), chunk_size))
	]

	del categories
	gc.collect()

	for chunk in tqdm(category_chunks):
		nodes_for_query = ", ".join(
			f"'{category}'" for category in chunk
		)

		results = table.search()\
			.where(f"category IN ({nodes_for_query})")\
			.limit(len(chunk))\
			.to_list()
		
		# NOTE:
		# Be sure to specify limit. Default limit (no .limit()) is 10.

		result_categories = [result["category"] for result in results]
		metadata += [
			category 
			for category in chunk 
			if category not in result_categories
		] # Build metadata list by adding categories that were NOT detected

		del nodes_for_query
		del results
		gc.collect()

	return metadata


def get_index(target: str, results: List[Dict[str, Any]]) -> int:
	'''
	Get the index of the target category from the vector DB results.
	@param: target (str), the target category.
	@param: results (List[Dict[str, Any]]), the results list from the 
		vector DB.
	@return: Returns the index of the target category from the resutls 
		list.
	'''
	# Iterate across every entry in the results list.
	for idx, result in enumerate(results):
		# Return the current index if the current category matches the 
		# target.
		if result["category"] == target:
			return idx
		
	# Return -1 if target category was not found.
	return -1


def bow_preprocess(text: str, return_word_freq: bool = False):
	'''
	Process the raw text to bag of words.
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
	# 2) handle special (circle) numbers 
	# 3) remove punctuation
	# 4) remove stop words
	# 5) remove superscripts/subscripts
	text = lowercase(text)
	text = handle_special_numbers(text)
	text = remove_punctuation(text)
	text = remove_stopwords(text)
	text = replace_subscripts(text)
	text = replace_superscripts(text)

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


def bow_chunk_on_stopwords(text: str, return_word_freq: bool = False):
	'''
	Process the raw text to bag of words.
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
	# 2) handle special (circle) numbers 
	# 3) remove punctuation
	# 4) remove stop words
	# 5) remove superscripts/subscripts
	text = lowercase(text)
	text = handle_special_numbers(text)
	text = remove_punctuation(text)
	# text = remove_stopwords(text)
	text = replace_subscripts(text)
	text = replace_superscripts(text)

	# Initialize stopwords set and regex pattern to filter out those
	# stop words.
	stop_words = set(stopwords.words("english"))
	pattern = r'\b(' + '|'.join(re.escape(word) for word in stop_words) + r')\b'
	
	# Apply the regex to remove the stopwords from the text. Filter out
	# any empty strings or stop words from the result.
	text_split = re.split(pattern, text)
	text_split = [
		segment.strip() 
		for segment in text_split 
		if segment.strip() and segment.lower() not in stop_words
	]
	bag_of_words = list(set(text_split))

	# Isolate the set of unique words in the remaining processed text.
	# bag_of_words = list(set(word_tokenize(text)))
	
	# Return just the bag of words if return_word_freq is False.
	if not return_word_freq:
		return tuple([bag_of_words])
	
 	# Record each word's frequency in the processed text.
	word_freqs = dict()
	words = word_tokenize(remove_stopwords(text))
	for word in bag_of_words:
		word_freqs[word] = words.count(word)

	# Return the bag of words and the word frequencies.
	return tuple([bag_of_words, word_freqs])


def rake_keyword_extraction(text: str):
	rake = Rake()
	rake.extract_keywords_from_text(text)
	keywords = rake.get_ranked_phrases()
	return keywords


def yake_keyword_extraction(text: str):
	kw_extractor = KeywordExtractor()
	keywords = kw_extractor.extract_keywords(text)
	keywords = [kw[0] for kw in keywords]
	return keywords


def textrank_keyword_extraction(text: str):
	# load a spaCy model, depending on language, scale, etc.
	nlp = spacy.load("en_core_web_sm")

	# add PyTextRank to the spaCy pipeline
	nlp.add_pipe("textrank")
	doc = nlp(text)
	keywords = [phrase for phrase in doc._.phrases]
	return keywords


def main():
	'''
	Main method. Process the word to document and document to word 
		metadata from their respective files to create word to IDF and
		document/article TF-IDF mappings for faster bag of words 
		processing during classical search (TF-IDF, BM25).
	@param: takes no arguments.
	@return: returns nothing.
	'''
	###################################################################
	# PROGRAM ARGUMENTS
	###################################################################
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--refresh_table",
		action="store_true",
		help="Whether to clear the table (if found in the DB) before adding the new data. Default is false/not specified."
	)
	parser.add_argument(
		"--num_thread",
		type=int,
		default=1,
		help="How many threads to use to process the data. Default is 1/not specified."
	)
	parser.add_argument(
		"--num_proc",
		type=int,
		default=1,
		help="How many processors to use to process the data. Default is 1/not specified."
	)
	parser.add_argument(
		"--use_json",
		action="store_true",
		help="Whether to use JSON or msgpack for loading metadata files. Default is false/not specified."
	)
	parser.add_argument(
		"--use_cpu",
		action="store_true",
		help="Whether to force machine to use CPU instead of detected GPU. Default is false/not specified."
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=1,
		help="Batch size for processing data to embedding model. Default is 1/not specified."
	)
	parser.add_argument(
		"--search_chunk_size",
		type=int,
		default=-1,
		help="The batch size for the number of texts that are going to be searched in the vector db. Default is -1/not specified."
	)
	parser.add_argument(
		"--run_tests",
		action="store_true",
		help="Whether to run some tests after all embeddings have been indexed and stored. Default is false/not specified."
	)

	# Parse arguments.
	args = parser.parse_args()

	# Unpack arguments.
	refresh_table = args.refresh_table
	use_json = args.use_json
	metadata_extension = "json" if use_json else "msgpack"
	use_cpu = args.use_cpu
	batch_size = args.batch_size
	run_tests = args.run_tests

	# Set random seed (for tests).
	random.seed(1234)

	# NOTE:
	# Be sure to process the category names for the table AND the final
	# graph or else the strings wont match.

	###################################################################
	# EMBEDDING MODEL SETUP
	###################################################################
	# Load the configurations from the config JSON.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load model dims to pass along to the schema init.
	model_name = config["vector-search_config"]["model"]
	dims = config["models"][model_name]["dims"]

	# Configure GPU for embeddings.
	device = "cpu"
	if not use_cpu:
		if torch.cuda.is_available():
			device = "cuda:0"

			# Clear cache from GPU.
			torch.cuda.empty_cache()
		elif torch.backends.mps.is_available():
			device = "mps"

	print(f"Running on device {device}")

	# NOTE:
	# Multiprocessing doesn't seem to play well with MPS device. If
	# using multiprocessing on Apple Silicon device, use the --use_cpu
	# flag to force the embedding model to run on CPU only.

	# Check for embedding model files and download them if necessary.
	tokenizer, model = load_model(config, device=device)

	###################################################################
	# INTIALIZE VECTOR DB
	###################################################################
	uri = "./data-lancedb/"
	db = lancedb.connect(uri, read_consistency_interval=timedelta(0))

	# Get list of tables.
	table_names = db.table_names()
	print(F"Tables in {uri}: {', '.join(table_names)}")

	# Search for specified table if it is available.
	table_name = f"category_table-model_{model_name}"
	table_in_db = table_name in table_names
	print(f"Searching for table '{table_name}' in database")
	print(f"Table found in database: {table_in_db}")

	# Initialize schema (this will be passed to the database when 
	# creating a new, empty table in the vector database).
	schema = pa.schema([
		pa.field("category", pa.utf8()),
		pa.field("vector", pa.list_(pa.float32(), dims))
	])

	# Create table if it does not exist in the database. Otherwise,
	# retrieve the table.
	if not table_in_db or (table_in_db and refresh_table):
		if table_in_db:
			db.drop_table(table_name)

		table = db.create_table(table_name, schema=schema)
	else:
		table = db.open_table(table_name)

	# Load categories from dataset.
	cat_to_doc_path = os.path.join(
		config["preprocessing"]["category_cache_path"],
		f"cat_to_doc.{metadata_extension}"
	)
	cat_to_docs = load_data_file(cat_to_doc_path, use_json)
	cat_to_docs = {
		preprocess_category_text(key): value
		for key, value in cat_to_docs.items()
	}
	all_categories = list(cat_to_docs.keys())
	all_categories = [
		preprocess_category_text(category) 
		for category in all_categories
		if len(preprocess_category_text(category).strip()) != 0
	]

	###################################################################
	# SEARCH FOR EXISTING EMBEDDINGS
	###################################################################
	
	# Filter out all categories that are already found in the table.
	print("Isolating categories already stored in table")

	# Resource Usage
	# ------------------------------
	# 1 processor @ batch size 256:
	# - 2.5 to 3 hours
	# - GB RAM
	# 16 processors @ batch size 16:
	# - 10 minutes
	# - 23 GB RAM

	# 1 processor
	#search_chunk_size = 10_000
	# - full database
	# - 36 GB RAM
	# - 45 minutes
	# - empty data
	# - 13 GB RAM
	# - 25 minutes
	# 16 processors
	#search_chunk_size = 1_000
	# - 44 GB RAM
	#search_chunk_size = 500
	# - 32 GB RAM
	#search_chunk_size = 100
	# - 23.5 GB RAM

	if args.search_chunk_size == -1:
		# Get the amount of RAM available to the system (in GB).
		memory_available = psutil.virtual_memory()[0] / (10 ** 9) 

		# Set the search chunk base size based on the amount of available
		# memory.
		if memory_available <= 8:
			search_chunk_base = 100
		elif memory_available <= 16:
			search_chunk_base = 500
		elif memory_available <= 32:
			search_chunk_base = 1_000
		else:
			search_chunk_base = 10_000
	else:
		# Set the search chunk size based on the user input.
		assert args.search_chunk_size > 0, \
			f"Argument --search_chunk_size is supposed to be > 0 if specified. Received {args.search_chunk_size}"
		search_chunk_base = args.search_chunk_size

	# With parallelization (Updated)
	# 11 GB RAM
	# 20 minutes empty table
	# 30 GB RAM
	# 9 hours full table

	# Full vector db with max search chunk base (10,000)
	# 16 processors (batch size 16)
	# 46 GB RAM (46 GB with subtraction/no change)
	# 1 hour
	# search chunk size = 10,000 / 16
	# 8 processors (batch size 32)
	# 43 GB RAM
	# 1.5 hours
	# search chunk size = 10,000 / 8
	# 4 processors (batch size 64)
	# 41 GB RAM
	# 2.5 hours
	# search chunk size = 10,000 / 4
	# 1 processor (batch size 256)
	# 40 GB RAM
	# 9 hours
	# search chunk size = 10,000 / 1

	# Full vector db with search chunk base (1,000)
	# 16 processors (batch size 16)
	# 23 GB RAM
	# 1.5 hours
	# search chunk size = 1,000 / 16
	# 8 processors (batch size 32)
	# 20 GB RAM
	# 3 hours
	# search chunk size = 1,000 / 8
	# 4 processors (batch size 64)
	# 19 GB RAM
	# 4.5 hours
	# search chunk size = 1,000 / 4
	# 1 processor (batch size 256)
	# 16.5 GB RAM
	# 17 hours
	# search chunk size = 1,000 / 1

	# NOTE:
	# Increased seearch chunk size means faster preprocessing with this
	# initial search in the vector DB at the cost of much higher RAM 
	# usage.
	remaining_categories = []

	divisor = args.num_proc if args.num_proc > 1 else args.num_thread
	chunk_size = math.ceil(len(all_categories) / divisor)
	search_chunk_size = math.ceil(search_chunk_base / divisor)
	category_node_chunks = [
		all_categories[i:i + chunk_size]
		for i in range(0, len(all_categories), chunk_size)
	]
	args_list = [
		(table, node_chunk, search_chunk_size) 
		for node_chunk in category_node_chunks
	]
	num_cpus = min(mp.cpu_count(), args.num_proc)
	
	if args.num_proc > 1:
		with mp.Pool(num_cpus) as pool:
			results = pool.starmap(
				search_table_for_categories, args_list
			)
		
			for result in results:
				remaining_categories += result
	else:
		with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
			results = executor.map(
				lambda args: search_table_for_categories(*args), 
				args_list
			)
		
			for result in results:
				remaining_categories += result

	print(f"Total number of categories: {len(all_categories)}")
	print(f"Number of categories left to embed: {len(remaining_categories)}")
	
	###################################################################
	# COMPUTE AND STORE REMAINING EMBEDDINGS
	###################################################################

	# NOTE:
	# Runtime on server
	# Single thread/processor: 12 hours (lowest RAM usage)
	# 12 threads: 9 hours
	# 12 processor: 4 hours (highest RAM usage)
	# "Maximized" processors on server:
	# 32 processors: 2.5 hours
	# 48 processors: 2.5 hours

	# NOTE:
	# (batch size > 1)batch size > 1:
	# 16 processors: 10 minutes (batch size 128)
	# 16 processors:  (batch size 32)
	# 8 processors @ batch size 16: 
	# - 8.5 hours for all embeddings
	# -  hours with no new embeddings
	# - 8.3 GB VRAM for all embeddings
	# -  GB VRAM with no new embeddings
	# -  GB RAM
	# - GPU status: pass
	# 8 processors @ batch size 32: 
	# -  hours for all embeddings
	# -  hours with no new embeddings
	# - 11.7 GB VRAM for all embeddings
	# -  GB VRAM with no new embeddings
	# -  GB RAM
	# - GPU status: pass
	# 16 processors @ batch size 16: 
	# - 8 hours for all embeddings
	# - 2.5 hours with no new embeddings
	# - 14.2 GB VRAM for all embeddings
	# - 5 GB VRAM with no new embeddings
	# - 18 GB RAM
	# - GPU status: pass
	# Anything bigger results in CUDA OOM on a single 16GB VRAM GPU on 
	# server. So no more than 256 items on the GPU at any time (for 
	# depth 10 graph).

	# With paralleization (Updated)
	# 16 x 16
	# 43 GB RAM
	# 14.2 GB VRAM
	# ~21 hours

	divisor = args.num_proc if args.num_proc > 1 else args.num_thread
	check_for_category = False
	chunk_size = math.ceil(len(remaining_categories) / divisor)
	write_chunk_size = 100_000
	category_node_chunks = [
		remaining_categories[i:i + write_chunk_size]
		for i in range(0, len(remaining_categories), write_chunk_size)
	]
	args_list = [
		(
			tokenizer, model, device, table, node_chunk, batch_size, 
			check_for_category
		)
		for node_chunk in category_node_chunks
	]

	print("Embedding categories to vectors and storing to table:")
	if args.num_proc > 1:
		with mp.Pool(num_cpus) as pool:
			for i in range(0, len(args_list), num_cpus):
				print(f"processing chunks {i + 1} to {i + num_cpus}")
				results = pool.starmap(
					embed_all_unseen_categories, args_list[i:i + num_cpus]
				)

				for result in results:
					table.add(
						data=result, 
						mode="append", 
						on_bad_vectors="drop"
					)

				del results
				gc.collect()
	else:
		with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
			for i in range(0, len(args_list), args.num_thread):
				print(f"processing chunks {i + 1} to {i + args.num_thread}")
				results = executor.map(
					lambda args: embed_all_unseen_categories(*args), 
					args_list[i:i + args.num_thread]
				)
			
				for result in results:
					table.add(
						data=result, 
						mode="append", 
						on_bad_vectors="drop"
					)

				del results
				gc.collect()

	# NOTE:
	# On server, fewer processors/threads means longer runtime but 
	# lower RAM/VRAM usage. Larger batch size scales RAM/VRAM usage
	# with the data but is also much faster.
	# - Low processors/threads, low batch size -> slowest but least
	# memory overhead
	# - High processors/threads, high batch size -> fastest but higher
	# memory overhead

	# Resource Usage
	# ------------------------------
	# 1 processor @ batch size 256:
	# - for all new embeddings
	#    -  hours
	#    - 13.6 GB VRAM
	# - for no new embeddings
	#    - hours
	# - GB RAM
	# 16 processors @ batch size 16:
	# - for all new embeddings
	#    - 22 hours
	#    - 14.6 GB VRAM
	# - for no new embeddings
	#    - hours
	#    - GB VRAM
	# - 38.7 GB RAM
	# NOTE:
	# Anything bigger results in CUDA OOM on a single 16GB VRAM GPU on 
	# server. So no more than 256 items on the GPU at any time (for 
	# depth 10 graph).

	# NOTE:
	# The following code here did not chunk the data in pieces of 100K
	# per thread. This would result in OOM issues when it came to ]
	# writing the embeddings to vector DB table. It has since been 
	# removed.

	if run_tests:
		# These tests are to measure how fast (and accurate) each index
		# is.
		print("Running tests on indices.")

		# Define accelerator based on device.
		# accelerator_device = device if device != "cpu" else None
		# print(f"Using accelerator device: {accelerator_device if accelerator_device else 'cpu'}")

		# Different indexes.
		flat_index = db.open_table(table_name)
		ivf_pq_index = db.open_table(table_name)
		ivf_pq_index.create_index(
			metric="cosine",
			vector_column_name="vector",
			# accelerator=accelerator_device
		)
		# hnsw_index = db.open_table(table_name)
		# hnsw_index.create_index(
		# 	metric="cosine",
		# 	vector_column_name="vector"
		# )

		# NOTE:
		# Despite the documentation for lancedb containing information
		# on HNSW indexes, it is only implemented in lancedb v0.17.X.

		# The number of options for k (the limit).
		k_options = [50, 100, 250, 500, 1_000]

		# Initialize list of queries.
		target_categories = []

		# Randomly sample categories.
		known_categories = random.sample(all_categories, 5)
		target_categories += known_categories

		# Generate some original queries that are not existing 
		# categories.
		original_queries = [
			"Who ran the 1936 track and field Olympic games for America?",			# Jesse Owens
			"What skyscraper in Dubai is the world's tallest structure?",	# Burj Khalifa
			"Name the fighter jet used in the movie Top Gun",				# F-14A Tomcat
			"What color are sapphires?",									# Blue/green/etc
			"Give an example of a shonen manga",							# Bleach/Dragon Ball/etc
		]

		# NOTE:
		# Raw queries like "What color are sapphires?" perform poorly.
		# Considering keyword extraction from raw queries and passing 
		# those keywords over to the search. This will cast a wider net
		# (per word) so limiting k may be a good idea given current 
		# retrieval speeds are quite good.

		# Combine all queries and generate the necessary embeddings.
		target_categories += original_queries
		target_category_embeddings = embed_text(
			tokenizer, model, device, target_categories
		)

		# Iterate through the target categories and embeddings.
		for idx, category in enumerate(target_categories):
			print(f"Target: {category}")
			embedding = target_category_embeddings[idx]

			# Try each option for top-k results.
			for k in k_options:
				print(f"Top {k} results:")

				# Query flat index for top K results.
				flat_start = time.time()
				flat_results = flat_index.search(embedding)\
					.limit(k)\
					.to_list()
				flat_end = time.time()
				flat_elapsed_time = flat_end - flat_start
				flat_target_index = get_index(category, flat_results)
				
				# Query IVF-PQ index for top K results.
				ivf_pq_start = time.time()
				ivf_pq_results = ivf_pq_index.search(embedding)\
					.limit(k)\
					.to_list()
				ivf_pq_end = time.time()
				ivf_pq_elapsed_time = ivf_pq_end - ivf_pq_start
				ivf_pq_target_index = get_index(category, ivf_pq_results)

				# Output results.
				print(f"Flat index:")
				print(f"\tquery time: {flat_elapsed_time:.4f}s")
				if category not in original_queries:
					print(f"\tposition: {flat_target_index}")
				print(f"\tTop 10 results:")
				print(
					json.dumps(
						[result["category"] for result in flat_results[:10]],
						indent=4
					)
				)
				print(f"IVF-PQ index:")
				print(f"\tquery time: {ivf_pq_elapsed_time:.4f}s")
				if category not in original_queries:
					print(f"\tposition: {ivf_pq_target_index}")
				print(f"\tTop 10 results:")
				print(
					json.dumps(
						[result["category"] for result in ivf_pq_results[:10]],
						indent=4
					)
				)
				print("-" * 32)
			
			print("=" * 72)

		# NOTE:
		# Basic keyword extraction (via the same bag-of-words methods 
		# used for my TF-IDF/BM25) did yield better outputs but there 
		# are visible areas where this could be improved by using
		# multiple words as keywords instead of single words (ie "color 
		# sapphires" vs "color", "sapphires").

		# NOTE:
		# Basic key phrase extraction (via the same bag-of-words methods
		# used for my TF-IDF/BM25) did yield even better outputs 
		# compared to previous basic key word extraction on unseen, raw
		# text queries. However, performance suffered for known exact-
		# matching categories. It may prove to be better overall to use
		# an ensemble of methods.

		# NOTE:
		# Keyword/keyphrase extraction via rake-nltk gave similar
		# outputs and performance to the key phrase extraction above.

		# NOTE:
		# Keyword/keyphrase extraction via yake gave better outputs for
		# unseen, raw text queries. Still struggles the same on known
		# exact-matching categories as above.

		# NOTE:
		# keyword/keyphrase extraction via scipy and pytextrank was not
		# able to work out of the box.

		# Iterate through the target categories and embeddings.
		for idx, category in enumerate(target_categories):
			print(f"Target: {category}")

			# Break down text into keywords.
			keywords = bow_preprocess(category)[0] # rough BOW to get key words.
			# keywords = bow_chunk_on_stopwords(category)[0] # rough key word extraction with BOW to get key phrases.
			# keywords = rake_keyword_extraction(category)
			# keywords = yake_keyword_extraction(category)
			# keywords = textrank_keyword_extraction(category)
			print(f"Target keywords: {', '.join(keywords)}")

			# TODO:
			# Extract keywords with alternative algorithsm and measure
			# the success. Algorithms to investigate include rake, 
			# yake, and textrank.

			# Generate embeddings.
			embeddings = embed_text(tokenizer, model, device, keywords)

			# Try each option for top-k results.
			for k in k_options:
				print(f"Top {k} results:")

				# Query flat index for top K results.
				flat_start = time.time()
				flat_results = []
				results_list = []
				for embedding in embeddings:
					results_list.append(flat_index.search(embedding)\
						.limit(k)\
						.to_list()
					)
				flat_end = time.time()

				# Round robin interspersing of results from each query.
				max_len = max(len(results) for results in results_list)
				for i in range(max_len):
					for results in results_list:
						if i < len(results):
							flat_results.append(results[i])

				flat_elapsed_time = flat_end - flat_start
				flat_target_index = get_index(category, flat_results)
				
				# Query IVF-PQ index for top K results.
				ivf_pq_start = time.time()
				ivf_pq_results = []
				results_list = []
				for embedding in embeddings:
					results_list.append(ivf_pq_index.search(embedding)\
						.limit(k)\
						.to_list()
					)
				ivf_pq_end = time.time()

				# Round robin interspersing of results from each query.
				max_len = max(len(results) for results in results_list)
				for i in range(max_len):
					for results in results_list:
						if i < len(results):
							ivf_pq_results.append(results[i])

				ivf_pq_elapsed_time = ivf_pq_end - ivf_pq_start
				ivf_pq_target_index = get_index(category, ivf_pq_results)

				# Output results.
				print(f"Flat index:")
				print(f"\tquery time: {flat_elapsed_time:.4f}s")
				if category not in original_queries:
					print(f"\tposition: {flat_target_index}")
				print(f"\tTop 10 results:")
				print(
					json.dumps(
						[result["category"] for result in flat_results[:10]],
						indent=4
					)
				)
				print(f"IVF-PQ index:")
				print(f"\tquery time: {ivf_pq_elapsed_time:.4f}s")
				if category not in original_queries:
					print(f"\tposition: {ivf_pq_target_index}")
				print(f"\tTop 10 results:")
				print(
					json.dumps(
						[result["category"] for result in ivf_pq_results[:10]],
						indent=4
					)
				)
				print("-" * 32)
			
			print("=" * 72)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()