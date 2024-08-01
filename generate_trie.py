# generate_trie.py
# Generate a trie (prefix tree) to map documents to individual words.
# Source: https://www.aleksandrhovhannisyan.com/blog/python-trie-data-structure/
# Python 3.9
# Windows/MacOS/Linux


import argparse
import gc
import json
import multiprocessing as mp
import os
import string
from typing import List, Dict, Any, Set, Tuple
# import unicodedata

import msgpack
from tqdm import tqdm
	

class TrieNodeGPT:
	def __init__(self) -> None:
		'''
		Initialize a trie node object.
		@param: Takes no arguments.
		@return: Returns an instance of the trie node object.
		'''
		# Initialize the children and document ids to an empty 
		# dictionary and set respectively.
		self.children = dict()
		self.document_ids = set()


	def __eq__(self, other: Any) -> bool:
		'''
		Determine if the object passed in/being compared to is 
			equivalent in both instance type and content.
		@param: other (Any), the object being compared to.
		@return: Returns a boolean from checking whether the input
			object is a trie node AND has the same content as the 
			current instance.
		'''
		# Check if the object is the same instance type.
		if not isinstance(other, TrieNodeGPT):
			return False

		# Check if children and document_ids are equal.
		return (self.children == other.children and
				self.document_ids == other.document_ids)


class TrieGPT:
	def __init__(self) -> None:
		'''
		Initialize a trie object.
		@param: Takes no arguments.
		@return: Returns an instance of the trie object.
		'''
		# Initialize a root node.
		self.root = TrieNodeGPT()


	def __eq__(self, other: Any) -> bool:
		'''
		Determine if the object passed in/being compared to is 
			equivalent in both instance type and content.
		@param: other (Any), the object being compared to.
		@return: Returns a boolean from checking whether the input
			object is a trie AND has the same content as the current
			instance.
		'''
		# Check if the object is the same instance type.
		if not isinstance(other, TrieGPT):
			return False
		
		# Check if the roots of both tries are equal (calls trie node
		# __eq__()).
		return self.root == other.root


	def insert(self, word: str, document_id: int) -> None:
		'''
		Insert a word and its associated document id into the trie.
		@param: word (str), the word that is to be inserted into the
			trie.
		@param: document_id (int), the document id associated to the 
			document where that word appears in, which will also be
			stored into the trie.
		@return: Returns nothing.
		'''
		# Set the pointer to the root of the tree.
		node = self.root

		# Iterate through each character in the word.
		for char in word:
			# If the character is not in the children, create a new
			# child node.
			if char not in node.children:
				node.children[char] = TrieNodeGPT()

			# Set the pointer to the child node.
			node = node.children[char]

		# NOTE:
		# According to the documentation, using add() inserts a single 
		# item vs update() inserts an iterable object (ie list).

		# Pointer is expected to be at the bottom most child given the
		# word. Add the document ID to the set in that node.
		node.document_ids.add(document_id)  # Original
		# node.document_ids.update(document_id)
	

	def search(self, word: str) -> Set[int] | None:
		'''
		Search for a word in the trie.
		@param: word (int), the word we want to query from the trie.
		@return: Returns either the set of document ids associated with
			the query word OR None if the word does not exist within 
			the trie.
		'''
		# Set the pointer to the root of the tree.
		node = self.root

		# Iterate through each character in the word.
		for char in word:
			# If the character is not in the children, return None.
			if char not in node.children:
				return None
			
			# Set the pointer to the child node.
			node = node.children[char]

		# Return None if the pointer is not a node, otherwise return
		# the document ids at the node.
		return node.document_ids if node else None
	

def serialize_trie_node(node: TrieNodeGPT) -> Dict:
	'''
	Serialize a trie node to a dictionary.
	@param: node (TrieNodeGPT), the trie node that needs to be 
		serialized into a dictionary.
	@return: Returns the trie node serialized as a dictionary.
	'''
	# Return a dictionary that recursilvely serializes the child nodes
	# and stores the current node's document ids.
	return {
		'children': {
			char: serialize_trie_node(child) 
			for char, child in node.children.items()
		},
		'document_ids': list(node.document_ids)
	}


def deserialize_trie_node(data: Dict) -> TrieNodeGPT:
	'''
	Deserialize a trie from a (loaded) dictionary into a trie node.
	@param: data (Dict), the trie data in the form of a dictionary
		(usually loaded from a file).
	@return: Returns a trie node containing all the appropriate object
		data.
	'''
	# Initialize a new node.
	node = TrieNodeGPT()

	# Set the document_ids portion of the node to the set of document 
	# ids from the current data passed in.
	node.document_ids = set(data['document_ids'])

	# Recursively retrieve the same deserialized data from the child
	# nodes and insert that into the current node's children.
	node.children = {
		char: deserialize_trie_node(child) 
		for char, child in data['children'].items()
	}

	# Return the current node.
	return node


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
	@return: Returns the trie loaded from the file.
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


def save_trie(trie: TrieGPT, path: str, use_json: bool = False) -> None:
	'''
	Save a trie to a file.
	@param: trie (TrieGPT), the trie that is to be stored.
	@param: path (str), the path of the file where the trie is to be 
		stored.
	@param: use_json (bool), whether the file being loaded is a JSON or
		msgpack. Default is False (msgpack).
	@return: Returns the trie loaded from the file.
	'''
	# Serialize the trie.
	trie_dict = serialize_trie_node(trie.root)

	# Write the serialized trie data to the file.
	write_data_file(path, trie_dict, use_json)


def load_trie(path: str, use_json: bool = False) -> TrieGPT:
	'''
	Load a trie from a file.
	@param: path (str), the path of the file where the trie is stored.
	@param: use_json (bool), whether the file being loaded is a JSON or
		msgpack. Default is False (msgpack).
	@return: Returns the trie loaded from the file.
	'''
	# Initialize an empty trie object.
	trie = TrieGPT()

	# Load the trie data from the file.
	trie_data = load_data_file(path, use_json)

	# Deserialize the trie data and load it into the trie.
	trie.root = deserialize_trie_node(trie_data)

	# Return the trie.
	return trie


def build_trie(limit: int, char: str, d2w_files: List[str], doc_to_int: Dict[str, int], 
			redirect_files: List[str], use_json: bool = False) -> TrieGPT:
	'''
	Construct a trie given a starting character and the list of 
		document to word files. The trie will be built from all words 
		in the corpus that start with that input character.
	@param: limit (int), the maximum length of a word that will be 
		stored in the trie.
	@param: char (str), the character that all strings in the trie are
		expected to start with.
	@param: d2w_files (List[str]), the list of all document to word 
		filepaths in the corpus.
	@param: doc_to_int (Dict[str, int]), the mapping of each 
		document/article to a unique numerical number.
	@param: redirect_files (List[str]), the list of all filepaths 
		containing the documents that redirect to other documents.
	@param: use_json (bool), whether to load the data file using JSON 
		msgpack (default is False).
	@return: Returns a TrieGPT object built from the corpus where the
		root node is the starting character passed in.
	'''
	# List of all common english alphanumerics.
	alpha_numerics = string.digits + string.ascii_lowercase

	# Initialize the trie for the character group.
	char_trie = TrieGPT()

	# Load up all redirects into a list.
	redirects = list()
	for redirect_file in redirect_files:
		redirects += load_data_file(
			redirect_file, use_json
		)

	# Iterate through the document to word files.
	for file in tqdm(d2w_files):
		# Load the doc to word data.
		doc_to_words = load_data_file(file, use_json)

		# Create a subset of all documents in the data that are 
		# redirects.
		docs = doc_to_words.keys()
		redirect_subset = set(redirects).intersection(set(docs))

		# Iterate through each document in the file.
		for doc in list(docs):
			# Skip the document if it is marked as a known redirect
			# document/article.
			if doc in redirect_subset:
				continue
	
			# Load the document word frequencies.
			word_freq = doc_to_words[doc]

			# Retrieve the document's numerical ID from the 
			# document to int dictionary.
			doc_id = doc_to_int[doc]

			# Iterate through each word in the word frequency map.
			for word in list(word_freq.keys()):
				# Isolate the first character of the word.
				word_char = word[0]

				# Skip words (do nothing/no insertions) that are too 
				# long (outside of the limit).
				if len(word) <= limit:
					# Insert words to the character trie if either:
					# 1) The target character is "other" and the 
					# current character is not in the alphanumerics.
					# 2) The target character is in the alphanumerics 
					# and the target character matches the current 
					# character.
					if char == "other" and word_char not in alpha_numerics:
						char_trie.insert(word, doc_id)
					elif word_char == char:
						char_trie.insert(word, doc_id)

	# Return the character trie.
	return char_trie


def explore_data() -> None:
	'''
	Perform a bit of data exploration by examining the number of words 
		that start with different characters.
	@param: Takes no arguments.
	@return: Returns nothing.
	'''
	# Load config.
	with open("config.json", "r") as f:
		config = json.load(f)

	# IDF path.
	idf_path = config["preprocessing"]["idf_cache_path"]
	extension = ".msgpack"

	# IDF files.
	idf_files = [
		os.path.join(idf_path, file) 
		for file in os.listdir(idf_path)
		if file.endswith(extension)
	]

	# Load IDF data.
	word_idf = dict()
	for file in idf_files:
		idf_data = load_data_file(file, False)
		word_idf.update(idf_data)

	# Isolate the words.
	words = list(word_idf.keys())
	del word_idf
	gc.collect()

	starting_chars = sorted(list(set([word[0] for word in words])))
	print(f"Detected {len(starting_chars)} unique starting characters in the corpus.")

	# Set character limit to eliminate ridiculously long strings that 
	# are probably not actual english words. Should help counter max-
	# recursion limit error too.
	limit = 60 # Limit was determined based on longest word in English language at 45 characters (Google'd it) but I allowed for some extra space.

	digit_count = 0
	alpha_count = 0
	print(f"Number of total words: {len(words)}")

	# NOTE:
	# Given that our text preprocessing lowercased all words, there 
	# should be no uppercase english alphabetical characters (covered
	# by string.ascii_uppercase). We confirmed this by checking the 
	# number of characters starting with ascii_letters including and
	# without the uppercase characters. Including uppercase characters
	# had no affect, hence the confirmation.

	starting_chars = string.digits + string.ascii_lowercase #+ string.ascii_uppercase
	for char in tqdm(starting_chars):
		select_words = [
			word for word in words 
			if word.startswith(char) and len(word) < limit
		]
		print(f"{char} word count: {len(select_words)}")
		if char.isdigit():
			digit_count += len(select_words)
		elif char.isalpha():
			alpha_count += len(select_words)
		else:
			continue

	print(f"Number of digit words: {digit_count}")
	print(f"Number of alpha words (lowercase & uppercase): {alpha_count}")

	# NOTE:
	# Unicodedata references: 
	# https://www.fileformat.info/info/unicode/category/index.htm
	# https://docs.python.org/3.10/library/unicodedata.html
	# https://www.ssec.wisc.edu/~tomw/java/unicode.html

	# NOTE:
	# Notes on running this experiment.
	# - Server OOMed (with 67 GB of RAM) when building the single trie 
	#	with all words at around 20,000,000 words (roughly half of
	#	corpus vocab). Shows that we have to initialize and build each
	#   trie for each starting character individually.
	# - Giving each starting character its own trie would create around
	#   21,000 files. But when checking the basic English 
	#   alphanumerical characters, they have roughly 88% coverage of 
	#   all words in the corpus, leaving only around 5 million words
	#   unaccounted for. For this reason, Each english alphanumerical
	#   character will get its own trie (and file) and the rest will be
	#   put in a separate trie.
	# - It takes around 15 minutes on server to build mapping of 
	#	documents to their document ids.
	# - It takes around 25 to 30 minutes on server to build and save a 
	#   trie for each alphanumerical starting character. This increased
	#   to 30 to 35 minutes when the code remove redirect documents was
	#   added.
	# - RAM overhead is around 35GB for just a single trie so using
	#   multprocessing (even on the server) is not advised without 
	#   sufficient memory resources.

	return


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
	# parser.add_argument(
	# 	"--restart",
	# 	action="store_true",
	# 	help="Specify whether to restart the preprocessing from scratch. Default is false/not specified."
	# )
	# parser.add_argument(
	# 	'--num_proc', 
	# 	type=int, 
	# 	default=1, 
	# 	help="Number of processor cores to use for multiprocessing. Default is 1."
	# )
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
	# EXPLORE DATA
	###################################################################
	# Comment out if you don't want to run.
	# explore_data()
	# exit()

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
	trie_path = config["preprocessing"]["trie_cache_path"]
	redirect_path = config["preprocessing"]["redirect_cache_path"]

	# IDF path.
	idf_path = config["preprocessing"]["idf_cache_path"]

	# IDF files.
	idf_files = [
		os.path.join(idf_path, file) 
		for file in os.listdir(idf_path)
		if file.endswith(extension)
	]

	# Load IDF data.
	word_idf = dict()
	for file in idf_files:
		idf_data = load_data_file(file, False)
		word_idf.update(idf_data)

	# Isolate the words.
	limit = 60 # Limit was determined based on longest word in English language at 45 characters (Google'd it) but I allowed for some extra space.
	words = [
		word for word in list(word_idf.keys())
		if len(word) <= limit	
	]

	# Clear up memory.
	del word_idf
	gc.collect()

	# Create trie cache path if it doesn't already exist.
	if not os.path.exists(trie_path):
		os.makedirs(trie_path, exist_ok=True)

	# Doc2Word files.
	d2w_files = sorted([
		os.path.join(d2w_path, file) 
		for file in os.listdir(d2w_path)
		if file.endswith(extension)
	])

	# Redirect files.
	redirect_files = [
		os.path.join(redirect_path, file) 
		for file in os.listdir(redirect_path)
		if file.endswith(extension)
	]

	# Set paths for document to document_id and the inverse map paths.
	doc_to_int_path = os.path.join(trie_path, "doc_to_int" + extension)
	int_to_doc_path = os.path.join(trie_path, "int_to_doc" + extension)

	###################################################################
	# BUILD/LOAD DOCUMENT IDS
	###################################################################
	# NOTE:
	# We have int_to_doc_str because msgpack is not able to read int
	# values as keys in dictionaries. We convert back to int when
	# loading from the file.

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
	# BUILD TRIES (SHARDED)
	###################################################################

	# NOTE:
	# Abandoned this section. Building tries from scratch while 
	# sharding is incredibly time and compute intensive. It would be
	# much more economical to build the full sized tries (per starting
	# letter) first before moving on and sharding each trie.

	###################################################################
	# BUILD TRIES
	###################################################################
	# Set character limit to eliminate ridiculously long strings that 
	# are probably not actual english words. Should help counter max-
	# recursion limit error too.
	limit = 60 # Limit was determined based on longest word in English language at 45 characters (Google'd it) but I allowed for some extra space.

	# Initialize a list of all common english alphanumerics.
	alpha_numerics = string.digits + string.ascii_lowercase
	print("Creating tries...")

	# Iterate through all alphanumerics and an "other" category. This
	# will serve as our target characters to build our tries.
	for char in list(alpha_numerics) + ["other"]:
		plural = "" if char in alpha_numerics else "s"
		print(f"Processing words that start with {char} character{plural}")
		# path = os.path.join(trie_path, char + "_trie" + extension)
		path = os.path.join(trie_path, char + "_trie_slim" + extension)

		# Skip if path exists.
		if os.path.exists(path):
			continue

		# Initialize and build the character trie.
		char_trie = build_trie(
			limit, char, d2w_files, doc_to_int, redirect_files, args.use_json
		)

		# Save the trie.
		# trie_dict = serialize_trie_node(char_trie.root)
		# print(f"Saving trie to {path}")
		# write_data_file(path, trie_dict, args.use_json)
		print(f"Saving trie to {path}")
		save_trie(char_trie, path, args.use_json)

		# Delete trie and collect garbage.
		del char_trie
		gc.collect()

	###################################################################
	# LOAD A TRIE
	###################################################################

	# Path to a test trie saved.
	# load_path = os.path.join(trie_path, "a_trie" + extension)
	load_path = os.path.join(trie_path, "a_trie_slim" + extension)
	# save_path = os.path.join(trie_path, "0_trie_copy" + extension)
	
	# Load the trie.
	print(f"Loading trie from {load_path}...")
	loaded_trie = load_trie(load_path, args.use_json)

	# Recompute the trie.
	print(f"Recomputing the same trie...")
	computed_trie = build_trie(
		limit, "a", d2w_files, doc_to_int, redirect_files, args.use_json
	)

	# Verify that the loaded trie and recomputed trie are equal.
	print(f"Loaded trie matched original: {loaded_trie == computed_trie}")

	# Delete trie and collect garbage.
	del computed_trie
	gc.collect()

	###################################################################
	# SEARCH A TRIE
	###################################################################

	# Run a test search on the loaded tries.
	search_terms = [
		"apple", "ambrosia", "asymptomatic", "approximatedly", "bee"
	] # "approximatedly" and "bee" should return nothing/None.

	# Iterate through each search query.
	for word in search_terms:
		# Search the trie.
		search_result = loaded_trie.search(word)

		# Print the results.
		print(f"Searching for : {word}")
		print("Recieved:")
		if search_result is None:
			# If results are None, just print the results/None.
			print(search_result)
		else:
			# If results are not None, iterate through each result and
			# print the results in the "document_id, document" format.
			results = list(search_result)
			for result in results:
				if isinstance(result, int):
					print(f"{result}, {int_to_doc[result]}")

	# Delete trie and collect garbage.
	del loaded_trie
	gc.collect()

	###################################################################
	# SHARD TRIES
	###################################################################

	# Set a chunk size.
	chunk_size = 100_000

	# Initialize a map for each shard to a range of values.
	shard_map = dict()

	# Iterate through all alphanumerics and "other" category.
	for char in list(alpha_numerics) + ["other"]:
		# Load character trie.
		load_path = os.path.join(
			trie_path, f"{char}_trie_slim" + extension
		)
		loaded_trie = load_trie(load_path, args.use_json)
		print(f"Sharding trie {load_path}")

		# Given the list of words, isolate and sort all (valid) words 
		# that start with the appropriate character.
		if char in list(alpha_numerics):
			char_words = [
				word for word in words
				if word[0] in alpha_numerics and len(word) <= limit
			]
		else:
			char_words = [
				word for word in words
				if word[0] not in alpha_numerics and len(word) <= limit
			]
		char_words = sorted(char_words)
		valid_char_words = list()

		# Remove words that do not appear in the trie.
		for word in char_words:
			results = loaded_trie.search(word)
			if results is not None:
				valid_char_words.append(word)

		# Clear up memory.
		del char_words
		gc.collect()

		# Sort again.
		valid_char_words = sorted(valid_char_words)
		print(f"Found {len(valid_char_words)} valid words for {char}")

		# Chunk the word list.
		chunk_size = 100_000
		word_chunks = [
			valid_char_words[i:i + chunk_size]
			for i in range(0, len(valid_char_words), chunk_size)
		]

		# Iterate through each word list chunk.
		for idx, chunk in enumerate(tqdm(word_chunks)):
			# Initialize a trie for the shard
			char_trie = TrieGPT()
			save_path = os.path.join(
				trie_path, f"{char}_shard_{idx + 1}_trie_slim" + extension
			)

			# Update shard map.
			shard_map[save_path] = [chunk[0], chunk[-1]]

			for word in chunk:
				# Match each word with the list of associated document 
				# IDs.
				document_ids = loaded_trie.search(word)
				assert document_ids is not None,\
					f"Search for {word} in {load_path} yielded no results"
				
				# if document_ids is None:
				# 	continue

				# Insert each word, document ID pair into the shard 
				# trie.
				for doc_id in list(document_ids):
					char_trie.insert(word, doc_id)

			# Save the shard trie.
			save_trie(char_trie, save_path, args.use_json)

			# Clear up memory.
			del char_trie
			gc.collect()

		# Clear up memory.
		gc.collect()

	# Save the shard map.
	shard_path = os.path.join(trie_path, "shard_map" + extension)
	write_data_file(shard_path, args.use_json)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()