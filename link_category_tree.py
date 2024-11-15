# link_category_tree.py
# Link all categories from the document dump over to the category tree
# downloaded from Wikipedia.
# Python 3.9
# Windows/MacOS/Linux


import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import timedelta
import json
import gc
import math
import multiprocessing as mp
import os
import pyarrow as pa
from typing import Dict, List, Set, Any

import lancedb
import msgpack
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from preprocess import load_model
import rust_search_helpers as rsh


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


def save_graph(G: nx.DiGraph, file_name: str, format: str = "graphml") -> None:
	'''
	Save the graph to a file.
	@param: G (nx.DiGraph), the graph of the category tree that is to 
		be saved.
	@param: file_name (str), the filename to save the graph as.
	@param: format (str), how the graph should be saved. Default is
		"graphml".
	@return: returns nothing.
	'''
	assert file_name.endswith(format),\
		f"Expected filename {file_name} to end with appropriate extension. Recieved {format}"

	valid_extensions = ["graphml", "gml", "edgelist", "json", "msgpack"]

	if format == "graphml":
		nx.write_graphml(G, file_name)  # Save as GraphML
	elif format == "gml":
		nx.write_gml(G, file_name)      # Save as GML
	elif format == "edgelist":
		nx.write_edgelist(G, file_name) # Save as Edge List
	elif format == "json":
		with open(file_name, "w+") as f:
			json.dump(nx.to_dict_of_lists(G), f, indent=4)
			# json.dump(nx.node_link_data(G), f, indent=4)
	elif format == "msgpack":
		write_data_to_msgpack(file_name, nx.to_dict_of_lists(G))
		# write_data_to_msgpack(file_name, nx.node_link_data(G))
	else:
		raise ValueError(f"Unsupported format: choose {', '.join(valid_extensions)}.")


def load_graph(file_name: str, format: str = "graphml") -> nx.DiGraph:
	'''
	Load the graph from a file.
	@param: file_name (str), the filename to load the graph from.
	@param: format (str), how the graph should was saved. Default is
		"graphml".
	@return: returns  the graph of the category tree that is was to be
		loaded.
	'''
	assert file_name.endswith(format),\
		f"Expected filename {file_name} to end with appropriate extension. Recieved {format}"

	valid_extensions = ["graphml", "gml", "edgelist", "json", "msgpack"]

	if format == "graphml":
		graph = nx.read_graphml(file_name)
	elif format == "gml":
		graph = nx.read_gml(file_name)
	elif format == "edgelist":
		graph = nx.read_edgelist(file_name)
	elif format == "json":
		with open(file_name, "r") as f:
			graph = nx.from_dict_of_lists(json.load(f))
			# graph = nx.node_link_graph(json.load(f), directed=True)
	elif format == "msgpack":
		graph = nx.from_dict_of_lists(load_data_from_msgpack(file_name))
		# graph = nx.node_link_graph(load_data_from_msgpack(file_name))
	else:
		raise ValueError(f"Unsupported format: choose {', '.join(valid_extensions)}.")
	
	return graph.to_directed()


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
	table: lancedb.table, 
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
	@param: table (lacedeb.table), the vector DB table that is being 
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
	local_limit = int(100_000 / batch_size)
	local_counter = 0
	# append_to_chunk = len(categories) > local_limit
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

		# Insert local data into the table once the local counter for
		# number of batches has reached the set threshold.
		# if local_counter > 0 and local_counter % local_limit == 0 and append_to_chunk and len(local_data) > 0:
		# 	table.add(data=local_data, mode="append") 
		# 	local_data = []

	# Insert any remaining local data into the table.
	# if len(local_data) > 0 and append_to_chunk:
	# 	table.add(data=local_data, mode="append")

	# Return the list of all category, embedding pairs computed (if the
	# number of categories embedded did not exceed the set threshold).
	return metadata


def search_table_for_categories(table: lancedb.table, categories: List[str], chunk_size: int = 100) -> List[str]:
	'''
	Search the categories in the vector DB table and isolate which 
		categories in the arguments list are already found in the 
		table.
	@param: table (lacedeb.table), the vector DB table that is being 
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
			.to_list()

		if len(results) > 0:
			metadata += [result["category"] for result in results]
			# print(json.dumps(metadata, indent=4))
			# exit()

		del nodes_for_query
		del results
		gc.collect()

	return metadata

	# print(len(results))
	# print(results[0])
	# exit()

	# for node in tqdm(categories):
	# 	# Query the vector DB for the category.
	# 	results = table.search()\
	# 		.where(f"category = '{node}'")\
	# 		.limit(1)\
	# 		.to_list()

	# 	# If the category does exist already, skip the entry.
	# 	if len(results) != 0:
	# 		metadata.append(node)
	
	# return metadata


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
		"--depth",
		type=int, 
		default=2, 
		help="Max level of depth to go in the wikipedia category tree. Default is 2."
	)
	parser.add_argument(
		"--extension",
		type=str, 
		default="graphml", 
		help="The file extension that should be used to save/load the graph. Default is 'graphml'."
	)
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

	# Parse arguments.
	args = parser.parse_args()

	# Unpack arguments.
	depth = args.depth
	extension = args.extension
	refresh_table = args.refresh_table
	use_json = args.use_json
	metadata_extension = "json" if use_json else "msgpack"
	use_cpu = args.use_cpu
	batch_size = args.batch_size

	# Isolate downloaded graph and verify its path.
	downloaded_graph_path = os.path.join(
		"./wiki_category_graphs",
		f"wiki_categories_depth{depth}.{extension}"
	)
	if not os.path.exists(downloaded_graph_path):
		print(f"Error: Could not find graph at {downloaded_graph_path}")
		exit(1)

	# Load the downloaded graph.
	downloaded_graph = load_graph(downloaded_graph_path, extension)
	downloaded_graph_nodes = [
		preprocess_category_text(node) 
		for node in list(downloaded_graph.nodes())
	]

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
			device = "cuda:1"

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
	# INTIALIZE VECTOR DB AND SEARCH FOR EXISTING EMBEDDINGS
	###################################################################
	uri = "./data-parallel-11_14_2024/lance_db"
	db = lancedb.connect(uri, read_consistency_interval=timedelta(0))

	# Get list of tables.
	table_names = db.table_names()
	print(F"Tables in {uri}: {', '.join(table_names)}")

	# Search for specified table if it is available.
	table_name = f"category_tree_table-depth_{depth}-model_{model_name}"
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

		# table = db.create_table(table_name, data=vector_metadata)
		table = db.create_table(table_name, schema=schema)
	else:
		table = db.open_table(table_name)

	# Load categories from dataset.
	cat_to_doc_path = os.path.join(
		config["preprocessing"]["category_cache_path"],
		f"cat_to_doc.{metadata_extension}"
	)
	cat_to_docs = load_data_file(cat_to_doc_path, use_json)
	missing_categories = rsh.set_difference(
		set(downloaded_graph_nodes), set(cat_to_docs.keys())
	)
	missing_categories = [
		preprocess_category_text(category) 
		for category in missing_categories
		if len(preprocess_category_text(category).strip()) != 0
	]
	all_categories = set(downloaded_graph_nodes + missing_categories)
	all_categories_list = list(copy.deepcopy(all_categories))

	# Filter out all categories that are already found in the table.
	print("Isolating categories already stored in table")
	found_categories = []

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

	divisor = args.num_proc if args.num_proc > 1 else args.num_thread
	chunk_size = math.ceil(len(all_categories) / divisor)
	search_chunk_size = 10_000
	category_node_chunks = [
		all_categories_list[i:i + chunk_size]
		for i in range(0, len(all_categories), chunk_size)
	]
	args_list = [
		(table, node_chunk, search_chunk_size) 
		for node_chunk in category_node_chunks
	]
	
	if args.num_proc > 1:
		with mp.Pool(min(mp.cpu_count(), args.num_proc)) as pool:
			results = pool.starmap(
				search_table_for_categories, args_list
			)

			for result in results:
				found_categories += result
	else:
		with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
			results = executor.map(
				lambda args: search_table_for_categories(*args), 
				args_list
			)
		
			for result in results:
				found_categories += result

	print(len(all_categories_list))
	print(len(found_categories))
	del all_categories_list
	all_categories.difference_update(found_categories)
	all_categories = list(all_categories)
	print(len(all_categories))
	# exit()
	
	###################################################################
	# COMPUTE AND STORE REMAINING EMBEDDINGS
	###################################################################
	vector_metadata = []

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

	divisor = args.num_proc if args.num_proc > 1 else args.num_thread
	check_for_category = False
	chunk_size = math.ceil(len(all_categories) / divisor)
	write_chunk_size = 100_000
	category_node_chunks = [
		# all_categories[i:i + chunk_size]
		# for i in range(0, len(all_categories), chunk_size)
		all_categories[i:i + write_chunk_size]
		for i in range(0, len(all_categories), write_chunk_size)
	]
	args_list = [
		(
			# tokenizer, model, device, table, node_chunk, batch_size, 
			# check_for_category
			tokenizer, model, device, table, node_chunk, batch_size, 
			check_for_category
		)
		for node_chunk in category_node_chunks
	]
	print("Embedding graph categories to vectors:")
	if args.num_proc > 1:
		num_cpus = min(mp.cpu_count(), args.num_proc)
		with mp.Pool(num_cpus) as pool:
			for i in range(0, len(args_list), num_cpus):
				print(f"processing chunks {i + 1} to {i + num_cpus}")
				results = pool.starmap(
					embed_all_unseen_categories, args_list[i:i + num_cpus]
				)

				for result in results:
					# vector_metadata += result
					table.add(
						data=result, 
						mode="append", 
						on_bad_vectors="drop"
					)
	else:
		with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
			results = executor.map(
				lambda args: embed_all_unseen_categories(*args), 
				args_list
			)
		
			for result in results:
				# vector_metadata += result
				table.add(
					data=result, 
					mode="append", 
					on_bad_vectors="drop"
				)

	
	exit()

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

	if args.num_proc > 1:
		with mp.Pool(min(mp.cpu_count(), args.num_proc)) as pool:
			results = pool.starmap(
				embed_all_unseen_categories, args_list
			)

			for result in results:
				vector_metadata += result
	else:
		with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
			results = executor.map(
				lambda args: embed_all_unseen_categories(*args), 
				args_list
			)
		
			for result in results:
				vector_metadata += result
	
	# Add the metadata.
	# if len(vector_metadata) != 0:
	# 	print(f"Adding {len(vector_metadata)} missing embeddings to vector DB.")
	# 	table.add(
	# 		data=vector_metadata, 
	# 		mode="append", 
	# 		on_bad_vectors="drop"
	# 	)

	exit()

	###################################################################
	# COMPUTE REMAINING CATEGORY EMBEDDINGS
	###################################################################

	# NOTE:
	# Seems to be CUDA OOM error when num_proc > 48 on server GPU (GPU 
	# detected/no --use_cpu flag).
	# "Maximized" processors on server:
	# 8 processors: (batch size 32)
	# 16 processors: 2.5 hours (batch size 64)
	# 16 processors:  minutes (batch size 128)
	# 32 processors: (batch size 1)
	# 48 processors: ~13 hours (batch size 1)
	# 8 processors @ batch size 16: 
	# -  hours for all embeddings
	# -  hours with no new embeddings
	# -  GB VRAM for all embeddings
	# -  GB VRAM with no new embeddings
	# -  GB RAM
	# - GPU status: pass
	# 8 processors @ batch size 32: 
	# -  hours for all embeddings
	# -  hours with no new embeddings
	# -  GB VRAM for all embeddings
	# -  GB VRAM with no new embeddings
	# -  GB RAM
	# - GPU status: pass
	# 16 processors @ batch size 16: 
	# - 12.5 hours for all embeddings
	# -  hours with no new embeddings
	# - 14.6 GB VRAM for all embeddings
	# -  GB VRAM with no new embeddings
	# - 37.4 GB RAM
	# - GPU status: pass
	# Anything bigger results in CUDA OOM on a single 16GB VRAM GPU on 
	# server. So no more than 256 items on the GPU at any time (for 
	# depth 10 graph).


	###################################################################
	# UPDATE GRAPH WITH MOST SIMILAR CATEGORIES AS CHILDREN
	###################################################################
	nodes_for_query = ", ".join(
		f"'{category}'" for category in tqdm(downloaded_graph_nodes)
	)
	print(nodes_for_query[:100])
	print(downloaded_graph_nodes[:5])
	print(list(cat_to_docs.keys())[:5])
	exit(0)
	for node in tqdm(missing_categories):
		# Get current category node embedding.
		_, embedding = table.search()\
			.where(f"category = '{node}'")\
			.limit(1)\
			.to_list()

		# Search for "closest" node embedding from the graph.
		best_category, _ = table.search(embedding)\
			.where(f"category IN ({nodes_for_query})")\
			.limit(1)\
			.to_list()

		# Create an edge in the downloaded graph connecting the "best
		# matching" category (from the original downloaded graph) to
		# the current category node from the "missing" categories list.
		downloaded_graph.add_edge(best_category, node)

	# Save the updated category tree graph.
	output_graph_path = os.path.join(
		"./wiki_category_graphs",
		f"wiki_categories_depth{depth}_full.{extension}"
	)
	save_graph(downloaded_graph, output_graph_path, extension)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()