# generate_category_tree.py
# Generate an inverted index that utilizes Wikipedia's category system
# to narrow down which documents to search through via key word 
# extraction with a modified TF-IDF. The expectation is that this 
# results in fewer documents being returned compared to a very simple
# word to document inverted index.
# Source: https://www.researchgate.net/publication/283227199_
#	A_method_for_automated_document_classification_using_
#	Wikipedia-derived_weighted_keywords
# Python 3.9
# Windows/MacOS/Linux


import argparse
import copy
import gc
import json
import math
import multiprocessing as mp
import os
from typing import List, Dict, Any, Set, Tuple

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
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


def index_documents_as_ints(d2w_files: List[str], use_json: bool = False) -> Dict[str, int]:
	'''
	Convert document/article strings in the corpus to unique int 
		values.
	@param: d2w_files (List[str]), the list of all document to word 
		filepaths in the corpus.
	@param: use_json (bool), whether the files being loaded are in a 
		JSON or msgpack. Default is False (msgpack).
	@return: Returns the mapping between documents and their unique int
		values (document IDs).
	'''
	# Initialize an empty dictionary for the document to int mappings
	# and the first int value to 0.
	doc_to_int = dict()
	int_value = 0

	# Iterate through all the document to word filepaths in the corpus.
	for file in tqdm(d2w_files):
		# Load the data from the current file.
		data = load_data_file(file, use_json)
		
		# Get the list of documents/articles from the keys. Iterate
		# through each document/article.
		articles = list(data.keys())
		for article in articles:
			# Map the document/article to the current int value and
			# increment the int value.
			doc_to_int[article] = int_value
			int_value += 1

	# Return the document to int mapping.
	return doc_to_int


def merge_mappings(results: List[List]) -> Tuple[Dict[str, Set[int]], Dict[str, Set[str]]]:
	aggr_cat_to_doc = dict()
	aggr_cat_to_cat = dict()

	for result in results:
		cat_to_doc, cat_to_cat = result

		for key, value in cat_to_doc.items():
			if key not in list(aggr_cat_to_doc.keys()):
				aggr_cat_to_doc[key] = value
			else:
				aggr_cat_to_doc[key].update(value)

		for key, value in cat_to_cat.items():
			if key not in list(aggr_cat_to_cat.keys()):
				aggr_cat_to_cat[key] = value
			else:
				aggr_cat_to_cat[key].update(value)
	
	return aggr_cat_to_doc, aggr_cat_to_cat


def multiprocess_index_categories(doc_files: List[str], doc_to_int: Dict[str, int], num_proc: int) -> Tuple[Dict[str, Set[int]], Dict[str, Set[str]]]:
	chunk_size = math.ceil(len(doc_files) / num_proc)
	doc_files_chunks = [
		doc_files[i:i + chunk_size]
		for i in range(0, len(doc_files), chunk_size)
	]

	arg_list = [
		[doc_files_chunk, doc_to_int] 
		for doc_files_chunk in doc_files_chunks
	]

	with mp.Pool(processes=num_proc) as pool:
		results = pool.starmap(
			index_categories_from_documents, arg_list
		)

		cat_to_doc, cat_to_cat = merge_mappings(results)
	
	return cat_to_doc, cat_to_cat


def index_categories_from_documents(doc_files: List[str], doc_to_int: Dict[str, int]) -> Tuple[Dict[str, Set[int]], Dict[str, Set[str]]]:
	'''
	Compute the mappings between categories and the document IDs as
		well as between the categories and different categories.
	@param: doc_files (List[str]), the list of all paths to the xml 
		article documents in the corpus.
	@param: doc_to_int (Dict[str, int]), the mapping of all documents
		(file hashes) to an unique integer value (document ID).
	@return: Returns a mapping between categories and document IDs and
		categories and other categories.
	'''
	# Initialize the mappings for category to category and category to
	# document (ID).
	cat_to_doc = dict()
	cat_to_cat = dict()

	# Iterate through each document in the dataset.
	for file in doc_files:
		print(f"Isolating categories from {os.path.basename(file)}...")

		# Load in file.
		with open(file, "r") as f:
			raw_data = f.read()

		# Parse file with beautifulsoup. Isolate the articles.
		soup = BeautifulSoup(raw_data, "lxml")
		pages = soup.find_all("page")

		# Iterate through each article.
		for page in tqdm(pages):
			# Isolate the article/page's SHA1.
			sha1_tag = page.find("sha1")

			# Skip articles that don't have a SHA1 (should not be 
			# possible but you never know).
			if sha1_tag is None:
				continue

			# Clean article SHA1 text.
			article_sha1 = sha1_tag.get_text()
			article_sha1 = article_sha1.replace(" ", "").replace("\n", "")

			# Isolate the article/page's redirect tag.
			redirect_tag = page.find("redirect")

			# Skip articles that have a redirect tag (they have no 
			# useful information in them).
			if redirect_tag is not None:
				continue

			# Compute the file hash.
			file_hash = file + article_sha1

			# Convert the file hash to the respective integer.
			document_id = doc_to_int[file_hash]

			# Extract the article text.
			title_tag = page.find("title")
			text_tag = page.find("text")

			# Verify that the title and text data is in the article.
			assert None not in [title_tag, text_tag]

			# Extract the text from each tag.
			title = title_tag.get_text()
			title = title.replace("\n", "").strip()
			text = text_tag.get_text().strip()

			# Isolate the categories form the text body.
			text_splits = text.split("\n")
			categories = list()

			# Split the text body and isolate the category lines. 
			# Append those categories to the list.
			for split in text_splits:
				if split.startswith("[[Category:") and split.endswith("]]"):
					category = split.replace("[[Category:", "")[:-2].strip()
					categories.append(category)

			# Update the respective mapping depending on the type of
			# article entry.
			if title.startswith("Category:"):
				# For articles that start with "Category:" in the 
				# title, take the extracted categories and map them to
				# the title category in the category to category map.
				category = title.replace("Category:", "")
				
				if category in list(cat_to_cat.keys()):
					cat_to_cat[category].update(categories)
				else:
					cat_to_cat[category] = set(categories)
			else:
				# For all other articles (regular articles with text),
				# Store the article's unique document ID to each 
				# category in the category to document (ID) map.
				for category in categories:
					if category in list(cat_to_doc.keys()):
						cat_to_doc[category].add(document_id)
					else:
						cat_to_doc[category] = set([document_id])

	# Return the category to document (ID) and category to category
	# maps.
	return cat_to_doc, cat_to_cat


def remove_cycles(cat_to_cat):
	'''
	Remove any cycles that appear in the category to category 
		map/graph.
	@param: cat_to_cat (Dict[str, Set[str]]) the mapping of different
		categories to other categories in the corpus.
	@return: Returns a mapping between categories and document IDs and
		categories and other categories.
	'''
	def dfs(node, visited, rec_stack, parent=None):
		'''
		Run DFS.
		@param: node (str), 
		@param: visited (Set[str]), 
		@param: rec_stack (Set[str]),
		@param: parent (str),
		@return: 
		'''
		visited.add(node)
		rec_stack.add(node)

		for neighbor in list(cat_to_cat.get(node, [])):
			if neighbor not in visited:
				if dfs(neighbor, visited, rec_stack, node):
					return True
			elif neighbor in rec_stack:
				cat_to_cat[node].remove(neighbor)

		rec_stack.remove(node)
		return False
	
	visited = set()
	rec_stack = set()

	# Isolate top-most nodes (no inbound vertices).
	top_categories = list(cat_to_cat.keys())
	remove_categories = list()
	for top_category in top_categories:
		# Check each list of outbound vertices for each node in the 
		# graph.
		for category in top_categories:
			if top_category in cat_to_cat[category]:
				remove_categories.append(category)
	remove_categories = list(set(remove_categories))
	root_categories = [
		category for category in top_categories
		if category not in remove_categories
	]

	# Iterate through each node in the graph and run DFS on each one to
	# handle any possible cycles.
	# for node in list(cat_to_cat.keys()):
	for node in root_categories:
		if node not in visited:
			dfs(node, visited, rec_stack)

	return cat_to_cat


def build_category_metadata(cat_to_cat, cat_to_doc, d2w_files, int_to_doc, use_json=False):


	cat_to_word_freq = dict()

	known_cat = list(cat_to_doc.keys())
	all_categories = list()
	for key in list(cat_to_cat.keys()):
		all_categories += [key] + list(cat_to_cat[key])

	empty_value = {
		"word_freq": dict(),
		"num_doc": 0,
	}
	
	for cat in all_categories:
		if cat not in known_cat:
			cat_to_word_freq[cat] = empty_value
		else:
			known_word_freq = dict()
			docs_ints = cat_to_doc[cat]
			for doc_int in docs_ints:
				doc = int_to_doc[doc_int]
				for file in d2w_files:
					data = load_data_file(file, use_json)
					if doc in list(data.keys()):
						word_freq = data[doc]
						for word in word_freq:
							if word in known_word_freq:
								known_word_freq[word] += word_freq[word]
							else:
								known_word_freq[word] = word_freq[word]

			cat_to_word_freq[cat] = {
				"word_freq": known_word_freq,
				"num_doc": len(docs_ints)
			}

	queue = all_categories
	while len(queue) != 0:
		# Pop the current category from the queue.
		category = queue.pop(0)

		# Isolate parent categories.
		parent_categories = list()
		for parent_category in list(cat_to_cat.keys()):
			if category in cat_to_cat[parent_category]:
				parent_categories.append(parent_category)

		# Identify child categories.
		child_categories = list()
		if category in list(cat_to_cat.keys()):
			child_categories = cat_to_cat[category]

		# Iterate throuch each child category.
		for child_category in child_categories:
			data = cat_to_word_freq[child_category]
			child_word_freq = data["word_freq"]
			current_word_freq = cat_to_word_freq[category]["word_freq"]

			# Update number of documents and word frequency parameter.
			cat_to_word_freq[category]["num_doc"] += data["num_doc"]
			for word in list(child_word_freq.keys()):
				if word in list(current_word_freq.keys()):
					cat_to_word_freq[category]["word_freq"][word] += word_freq[word]
				else:
					cat_to_word_freq[category]["word_freq"][word] = word_freq[word]

		# Insert parent categories into the queue.
		parent_categories = [
			cat_ for cat_ in parent_categories if cat_ not in queue
		]
		queue += parent_categories

	assert empty_value not in cat_to_word_freq.values()
	return cat_to_word_freq


def main():
	'''
	Main method. Process the documents to build a directed acyclic 
		graph organizing the categories associated with each document.
		Additionally, compute the category level TF-IDF metadata 
		necessary.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	###################################################################
	# PROGRAM ARGUMENTS
	###################################################################
	
	parser = argparse.ArgumentParser()
	# parser.add_argument(
	# 	"--restart",
	# 	action="store_true",
	# 	help="Specify whether to restart the preprocessing from scratch. Default is false/not specified."
	# )
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

	# TODO:
	# Add restart and multiprocessing functionality. Given the amount 
	# of memory single processing consumes, this is not an immediate
	# necessity.

	###################################################################
	# VERIFY METADATA FILES
	###################################################################

	# Open config file.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Set file extension.
	extension = ".json" if args.use_json else ".msgpack"

	# Load paths.
	d2w_path = config["preprocessing"]["doc_to_words_path"]
	category_path = config["preprocessing"]["category_cache_path"]

	# Create category cache path if it doesn't already exist.
	if not os.path.exists(category_path):
		os.makedirs(category_path, exist_ok=True)

	# Doc2Word files.
	d2w_files = sorted([
		os.path.join(d2w_path, file) 
		for file in os.listdir(d2w_path)
		if file.endswith(extension)
	])

	# Document files.
	file_dir = "./WikipediaEnDownload/WikipediaData"
	doc_files = [
		os.path.join(file_dir, file)
		for file in os.listdir(file_dir)
		if file.endswith(".xml")
	]

	###################################################################
	# BUILD/LOAD DOCUMENT IDS
	###################################################################

	# NOTE:
	# We have int_to_doc_str because msgpack is not able to read int
	# values as keys in dictionaries. We convert back to int when
	# loading from the file.

	# Set paths for document to document_id and the inverse map paths.
	doc_to_int_path = os.path.join(category_path, "doc_to_int" + extension)
	int_to_doc_path = os.path.join(category_path, "int_to_doc" + extension)

	# Load or initialize map from documents to unique IDs.
	if not os.path.exists(doc_to_int_path) or not os.path.exists(int_to_doc_path):
		print("Indexing all documents to unique numerical IDs...")
		doc_to_int = index_documents_as_ints(d2w_files, args.use_json)
		int_to_doc = {value: key for key, value in doc_to_int.items()}
		int_to_doc_str = {
			str(key): value for key, value in int_to_doc.items()
		}

		# Save to file.
		write_data_file(doc_to_int_path, doc_to_int, args.use_json)
		write_data_file(int_to_doc_path, int_to_doc_str, args.use_json)
	else:
		print("Loading all document to unique ID mappings...")
		doc_to_int = load_data_file(doc_to_int_path, args.use_json)
		int_to_doc = load_data_file(int_to_doc_path, args.use_json)
		int_to_doc = {
			int(key): value for key, value in int_to_doc.items()
		}

	# Verify document to document id map (and its inverse) is 
	# initialized.
	assert doc_to_int is not None
	assert int_to_doc is not None

	###################################################################
	# BUILD/LOAD CATEGORY TO DOCUMENT & CATEGORY MAPPPINGS
	###################################################################

	# Set paths for category to document (id) and category to category.
	cat_to_doc_path = os.path.join(category_path, "cat_to_doc" + extension)
	cat_to_cat_path = os.path.join(category_path, "cat_to_cat" + extension)

	if not os.path.exists(cat_to_doc_path) or not os.path.exists(cat_to_cat_path):
		print("Indexing all categories...")
		if args.num_proc > 1:
			max_proc = min(mp.cpu_count(), args.num_proc)
			
			cat_to_doc, cat_to_cat = multiprocess_index_categories(
				doc_files, doc_to_int, max_proc
			)
		else:
			cat_to_doc, cat_to_cat = index_categories_from_documents(
				doc_files, doc_to_int
			)
		cat_to_doc_str = {
			key: [str(val) for val in value] 
			for key, value in cat_to_doc.items()
		}
		cat_to_cat_str = {
			key: list(value) for key, value in cat_to_cat.items()
		}

		# Save to file.
		write_data_file(cat_to_doc_path, cat_to_doc_str, args.use_json)
		write_data_file(cat_to_cat_path, cat_to_cat_str, args.use_json)
	else:
		print("Loading all category mappings...")
		cat_to_doc = load_data_file(cat_to_doc_path, args.use_json)
		cat_to_cat = load_data_file(cat_to_cat_path, args.use_json)
		cat_to_doc = {
			key: set([int(val) for val in value])
			for key, value in cat_to_doc.items()
		}
		cat_to_cat = {
			key: set(value) for key, value in cat_to_cat.items()
		}

	print(f"Number of unique categories (1): {len(list(cat_to_doc.keys()))}") # 3648460
	print(f"Number of unique categories (2): {len(list(cat_to_cat.keys()))}") # 2346348
	all_categories = []
	for key in list(cat_to_cat.keys()):
		all_categories += [key] + list(cat_to_cat[key])
	print(f"Number of unique categories (3): {len(set(all_categories))}") # 4644198 <- accurate count

	# Remove cycles in the category to category mapping.
	cat_to_cat_new = copy.deepcopy(cat_to_cat)
	cat_to_cat_new = remove_cycles(cat_to_cat_new)
	print(cat_to_cat_new == cat_to_cat)

	###################################################################
	# BUILD/LOAD CATEGORY TO WORD FREQUENCY MAP
	###################################################################

	cat_word_freq = dict()
	for category in tqdm(cat_to_doc):
		# Get all documents.
		documents = [
			int_to_doc[document] 
			for document in cat_to_doc[category]
		]

		# Load document word frequencies 
		pass

	###################################################################
	# BUILD/LOAD CATEGORY TO CATEGORY GRAPH
	###################################################################


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()