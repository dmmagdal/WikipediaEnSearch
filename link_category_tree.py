# link_category_tree.py
# Link all categories from the document dump over to the category tree
# downloaded from Wikipedia.
# Python 3.9
# Windows/MacOS/Linux


import argparse
from concurrent.futures import ThreadPoolExecutor
import json
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
				return_tensors="pt"
			).to(device)
		)

		# Post process the embedding (take mean against middle dim, 
		# send to CPU, isolate remaining vector)
		embedding = output[0].mean(dim=1)
		embedding = embedding.to("cpu")
		embedding = embedding.numpy()[0]

	# Return embedding.
	return embedding


def embed_all_unseen_categories(tokenizer: AutoTokenizer, model: AutoModel, device: str, table: lancedb.table, categories: List[str]) -> List[Dict[str, Any]]:
	metadata = []

	for node in tqdm(categories):
		# Query the vector DB for the category.
		results = table.search()\
			.where(f"category = '{node}'")\
			.limit(1)\
			.to_list()

		# If the category does exist already, skip the entry.
		if len(results) != 0:
			continue

		# Embed the category and add it to the vector metadata.
		embedding = embed_text(tokenizer, model, device, node)
		metadata.append({"category": node, "vector": embedding})

	return metadata


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

	# Parse arguments.
	args = parser.parse_args()

	# Unpack arguments.
	depth = args.depth
	extension = args.extension
	refresh_table = args.refresh_table
	use_json = args.use_json
	metadata_extension = "json" if use_json else "msgpack"
	use_cpu = args.use_cpu

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
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	elif use_cpu:
		device = "cpu"

	# Check for embedding model files and download them if necessary.
	tokenizer, model = load_model(config, device=device)

	###################################################################
	# INTIALIZE VECTOR DB AND STORE DOWNLOAD GRAPH EMBEDDINGS
	###################################################################
	uri = "./data/lance_db"
	db = lancedb.connect(uri)

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
	
	###################################################################
	# COMPUTE DOWNLOAD GRAPH EMBEDDINGS
	###################################################################
	vector_metadata = []

	# NOTE:
	# Runtime on server
	# Single thread/processor: 12 hours (lowest RAM usage)
	# 12 threads: 9 hours
	# 12 processor: 4 hours (highest RAM usage)

	divisor = args.num_proc if args.num_proc > 1 else args.num_thread
	chunk_size = math.ceil(len(downloaded_graph_nodes) / divisor)
	graph_nodes_chunks = [
		downloaded_graph_nodes[i:i + chunk_size]
		for i in range(0, len(downloaded_graph_nodes), chunk_size)
	]
	args_list = [
		(tokenizer, model, device, table, node_chunk)
		for node_chunk in graph_nodes_chunks
	]
	print("Embedding downloaded graph categories to vectors:")

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

	# for node in tqdm(downloaded_graph_nodes):
	# 	# Query the vector DB for the category.
	# 	# print(node)
	# 	results = table.search()\
	# 		.where(f"category = '{node}'")\
	# 		.limit(1)\
	# 		.to_list()
	# 	# print(results)
	# 	# print(len(results))
	# 	# exit()
	#
	# 	# If the category does exist already, skip the entry.
	# 	if len(results) != 0:
	# 		continue
	#
	# 	# Embed the category and add it to the vector metadata.
	# 	embedding = embed_text(tokenizer, model, device, node)
	# 	vector_metadata.append({"category": node, "vector": embedding})
	# with torch.no_grad():
	# 	for node in tqdm(downloaded_graph_nodes):
	# 		output = model(
	# 			**tokenizer(
	# 				node,
	# 				add_special_tokens=False,
	# 				padding="max_length",
	# 				return_tensors="pt"
	# 			).to(device)
	# 		)
	# 
	# 		embedding = output[0].mean(dim=1)
	# 		embedding = embedding.to("cpu")
	# 		embedding = embedding.numpy()[0]
	#
	# 		vector_metadata.append(
	# 			{"category": node, "vector": embedding}
	# 		)

	# Add the metadata.
	if len(vector_metadata) != 0:
		print(len(vector_metadata))
		print("Adding missing embeddings to vector DB.")
		table.add(data=vector_metadata, mode="append")

	###################################################################
	# COMPUTE REMAINING CATEGORY EMBEDDINGS
	###################################################################
	new_vector_metadata = []

	# Load categories from dataset.
	cat_to_doc_path = os.path.join(
		config["preprocessing"]["category_cache_path"],
		f"cat_to_doc.{metadata_extension}"
	)
	cat_to_docs = load_data_file(cat_to_doc_path, use_json)
	missing_categories = rsh.set_difference(
		set(downloaded_graph_nodes), set(cat_to_docs.keys())
	)

	chunk_size = math.ceil(len(downloaded_graph_nodes) / divisor)
	category_chunks = [
		missing_categories[i:i + chunk_size]
		for i in range(0, len(missing_categories), chunk_size)
	]
	args_list = [
		(tokenizer, model, device, table, node_chunk)
		for node_chunk in category_chunks
	]
	print("Embedding remaining categories to vectors:")

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
	if len(new_vector_metadata) != 0:
		print(len(new_vector_metadata))
		print("Adding remaining missing embeddings to vector DB.")
		table.add(data=new_vector_metadata, mode="append")

	###################################################################
	# UPDATE GRAPH WITH MOST SIMILAR CATEGORIES AS CHILDREN
	###################################################################

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()