# fill_category_graph.py
# Script to fill in the blanks in the category tree for the remainder 
# of the categories that covers the remainder of the documents.
# Python 3.9
# Windows/MacOS/Linux


import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import json
import os
from typing import Dict, List, Set

import rust_search_helpers as rsh

import msgpack
import networkx as nx
from tqdm import tqdm
# from transformers import pipeline, AutoTokenizer, 


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


# Load the graph from a file
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


def is_doc_in_missed_docs():
	pass


def fill_in_category_coverage():
	pass


def main():
	# Initialize argument parser and arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--depth",
		type=int, 
		default=2, 
		help="Max level of depth to go in the wikipedia category tree. Default is 2."
	)
	parser.add_argument(
		"--filetype",
		type=str, 
		default="graphml", 
		help="Filetype of the category graph to query. Default is graphml."
	)
	parser.add_argument(
		"--use_json",
		action="store_true",
		help="Specify whether to load and write metadata to/from JSON files. Default is false/not specified."
	)

	# Parse arguments.
	args = parser.parse_args()

	# Isolate argument values.
	depth = args.depth
	format = args.filetype
	use_json = args.use_json
	extension = ".json" if use_json else ".msgpack"

	# Set up graph load and save paths.
	folder = "./wiki_category_graphs"
	load_path = os.path.join(
		folder,
		f"wiki_categories_depth{depth}.{format}"
	)
	save_path = os.path.join(
		folder,
		f"wiki_categories_depth{depth}_full.{format}"
	)
	graph = load_graph(load_path, format)

	###################################################################
	# IDENTIFY TARGET CATEGORIES
	###################################################################

	# Load config.
	with open ("config.json", "r") as f:
		config = json.load(f)

	category_folder = config["preprocessing"]["category_cache_path"]

	# Load category to documents mapping..
	cat2doc_path = os.path.join(category_folder, f"cat_to_doc{extension}")
	cat2doc = load_data_file(cat2doc_path, use_json)

	# Load missed categories and documents.
	with open("missed_categories.json") as mc_f:
		missed_cats = json.load(mc_f)

	with open("missed_documents.json", "r") as md_f:
		missed_docs = json.load(md_f)


	missed_docs_set = set(missed_docs)
	# cat2doc_filtered = rsh.filter_category_map(
	# 	cat2doc, missed_docs_set, missed_cats
	# )
	# rsh.verify_filtered_category_map(
	# 	cat2doc_filtered, missed_docs_set, missed_cats
	# )

	# NOTE:
	# Similar to the problems from KBAI course at GATech OMSCS. 
	# Treating this like a state space problem. Initial state will be
	# no solution with no coverage and the goal state is a solution
	# with full coverage of the remaining documents.

	# Initialize key variables.
	# solution = []
	# visited = []
	# coverage = 0
	# full_coverage = len(missed_docs)
	# initial_state = (missed_cats, coverage, solution)
	# queue = [initial_state]
	# is_solved = False

	# Iterate through a heavily modified BFS to find the smallest
	# combination of categories that would cover the remaining missed
	# documents from the dump.
	print("Isolatinging minimum number of categories for full coverage:")
	# while len(queue) != 0 and not is_solved:
	# 	# Pop the state from the queue and unpack it.
	# 	available_categories, document_coverage, current_solution = queue.pop(0)

	# 	# Check for document coverage. If we have 100% coverage, this
	# 	# is a sign that we have reached a solution state.
	# 	if document_coverage == full_coverage:
	# 		is_solved = True
	# 		solution = copy.deepcopy(current_solution)
	# 		continue

	# 	# Skip solutions (category combinations) that have been 
	# 	# visited. Convert the current solution to a set because order
	# 	# of the categories in each visited solution doesnt matter.
	# 	if set(current_solution) in visited:
	# 		continue

	# 	# Sort the list of available categories, giving preference to
	# 	# the ones that have more document coverage.
	# 	sorted_categories = sorted(
	# 		available_categories,
	# 		key=lambda category: len(cat2doc[category]),
	# 		reverse=True
	# 	)

	# 	# Iterate through each available category in the sorted list.
	# 	# Generate new possible states and append them to the queue.
	# 	options = []
	# 	for category in tqdm(sorted_categories):
	# 		# Initialize a new (hypothesis) solution by appending the
	# 		# current category to the end of the current solution.
	# 		new_solution = current_solution + [category]

	# 		# Compute the new solution's document coverage.
	# 		# covered_documents = set()
	# 		# for solution_category in new_solution:
	# 		# 	covered_documents.update(cat2doc[solution_category])
	# 		# for doc in covered_documents:
	# 		# 	if doc not in missed_docs:
	# 		# 		covered_documents.remove(doc)
	# 		# covered_documents = [
	# 		# 	doc 
	# 		# 	for solution_category in new_solution
	# 		# 	for doc in cat2doc[solution_category]
	# 		# 	if doc in missed_docs
	# 		# ]
	# 		# new_document_coverage = len(set(covered_documents))
	# 		covered_documents = set()
	# 		for solution_category in new_solution:
	# 			covered_documents.update(cat2doc[solution_category])	# Requires all documents for all (possible) missed categories be filtered (have only documents from missed documents list).
	# 		new_document_coverage = len(covered_documents)

	# 		# Skip appending states that do not increase the coverage.
	# 		if new_document_coverage <= document_coverage:
	# 			continue

	# 		# Remove the current category from the list of available 
	# 		# categories.
	# 		remaining_categories = copy.deepcopy(available_categories)
	# 		remaining_categories.remove(category)

	# 		# Create new state tuple and update the options list 
	# 		# accordingly. Also update the list of visited solutions 
	# 		# too.
	# 		new_state = (remaining_categories, new_document_coverage, new_solution)
	# 		# queue.append(new_state)
	# 		options.append(new_state)
	# 		visited.append(set(new_solution))
			
	# 		# Memory cleanup.
	# 		del new_solution
	# 		del covered_documents
	# 		del remaining_categories
	# 		del new_state
	# 		gc.collect()

	# 	# Sort list of new state options by highest coverage (priority 
	# 	# goes to solutions that offer higher document coverage). 
	# 	# Append the sorted list to the queue.
	# 	sorted_options = sorted(
	# 		options, key=lambda state: state[1], reverse=True
	# 	)
	# 	queue += sorted_options

	# 	# Memory cleanup.
	# 	del available_categories
	# 	del document_coverage
	# 	del current_solution
	# 	del sorted_categories
	# 	del options
	# 	gc.collect()

	# # Memory cleanup.
	# del queue
	# del visited
	# gc.collect()
	# solution = rsh.minimum_categories_for_coverage(
	# 	cat2doc, missed_docs_set, missed_cats, use_bfs=True
	# )
	# solution = rsh.minimum_categories_for_coverage(
	# 	cat2doc, missed_docs_set, missed_cats, use_bfs=False
	# )
	solution = rsh.minimum_categories_for_coverage_new(
		cat2doc, missed_docs_set, missed_cats
	)

	if not len(solution) == 0:
		print(f"No solution was found")
	else:
		print(f"Solution was found")

	with open("filler_categories.json", "w+") as f:
		json.dump(list(solution), f, indent=4)




	###################################################################
	# UPDATE CATEGORY TREE
	###################################################################

	# save_graph(graph, save_path, format)


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()