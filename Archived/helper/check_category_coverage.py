# check_category_coverage.py
# Run a check against the different downloaded category tree maps from 
# the Wikipedia API with the articles in the current dump. 


import argparse
import json
import os
from typing import Dict

import msgpack
import networkx as nx
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


def main():
	# Initialize argument parser and arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--depth",
		type=int, 
		default=2, 
		help="Max level of depth to go in the wikipedia category tree. Default is 2."
	)
	# Parse arguments.
	args = parser.parse_args()

	# Control the depth of subcategory exploration
	depth = args.depth

	# Load file if it exists.
	filename = f"wiki_categories_depth{depth}.graphml"
	if not os.path.exists(filename):
		print(f"Could not detect file {filename}")
		exit(1)

	# Isolate node list (categories)
	graph = nx.read_graphml(filename)
	node_list = list(graph.nodes)
	# graph_dict = graph.to_dict_of_dicts(graph)
	print(node_list[:5])

	# Load category to documents.
	cat_to_doc = load_data_file(
		"./metadata/bag_of_words/category_cache/cat_to_doc.msgpack",
		use_json=False
	)
	print(list(cat_to_doc.keys())[:5])

	# Initialize list and set containing the names of the missed 
	# categories and documents.
	missed_categories = list()
	missed_documents = set()

	# Iterate through the list of documents and update the set of 
	# missed documents to contain all documents in the dump.
	for doc in cat_to_doc.values():
		missed_documents.update(doc)

	# Iterate through the categories and documents in the category to
	# documents mapping.
	for category, documents in tqdm(list(cat_to_doc.items())):
		# If the category does not exist within the node list from the
		# wikipedia category tree, append that category to the missed
		# categories list. Otherwise, remove the respective documents
		# from the missed documents set.
		if "Category:" + category not in node_list:
			missed_categories.append(category)
		else:
			missed_documents.difference_update(documents)

	# Print out how many categories and documents were missed.
	print(f"Number of categories missed: {len(missed_categories)}")
	print(f'Number of documents missed: {len(missed_documents)}')

	# Save that list from both in the JSON files.
	with open("missed_categories.json", "w+") as f1:
		json.dump(missed_categories, f1, indent=4)
	with open("missed_documents.json", "w+") as f2:
		json.dump(list(missed_documents), f2, indent=4)

	# NOTE:
	# Running single core, this process takes about 36 hours on server.
	# Would highly recommend implementing some parallelization.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()