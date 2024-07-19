# generate_trie.py
# Generate a trie (prefix tree) to map documents to individual words.
# Source: https://www.aleksandrhovhannisyan.com/blog/python-trie-data-structure/
# Python 3.9
# Windows/MacOS/Linux


import argparse
import gc
import json
import os
import string
from typing import List, Dict
import unicodedata

import msgpack
from tqdm import tqdm


class TrieNode:
	def __init__(self, text="") -> None:
		self.text = text
		self.children = dict()
		self.is_word = False
		self.document_ids = set()
	

class Trie:
	def __init__(self) -> None:
		# Initialize root node of the tree.
		self.root = TrieNode()


	def insert(self, word):
		# Set pointer to the root of the tree.
		current = self.root

		# Iterate through each character of the word.
		for i, char in enumerate(word):
			# If the character of the word, isolate the prefix (slice
			# the word up to the current character) and create a new
			# node with that prefix and insert it to the child under
			# the current node's dictionary.
			if char not in current.children:
				prefix = word[0:i + 1]
				current.children[char] = TrieNode(prefix)

			# Move the pointer to the child node.
			current = current.children[char]

		# Set the current pointer to have the current node as a word.
		current.is_word = True


	def find(self, word):
		# Set pointer to the root of the tree.
		current = self.root

		# Iterate through each character in the word.
		for char in word:
			# If the character was not in the pointer node's 
			# dictionary, return None.
			if char not in current.children:
				return None
			
			# Set the current pointer to the child node of the next
			# character in the word in the dictionary.
			current = current.children[char]

		# If the current pointer node is indicative of a word, return
		# the current node.
		if current.is_word:
			return current
		
		# Return None if the current pointer was not indicative of a
		# word.
		return None


	def starts_with(self, prefix):
		# NOTE:
		# This function is not needed for the purposes of this project.
		# This is for searching for a list of possible words given a 
		# prefix text.

		# Initialize word list and set pointer to the root.
		words = list()
		current = self.root

		# Iterate through the character in the prefix.
		for char in prefix:
			# If the character is not in the pointer children, return
			# an empty list.
			if char not in current.children:
				# Could also just return words since it's empty by default
				return list()
			
			# Set the pointer to the child node.
			current = current.children[char]

		self.__child_words_for(current, words)
		return words


	def __child_words_for(self, node, words):
		'''
		Private helper function. Cycles through all children
		of node recursively, adding them to words if they
		constitute whole words (as opposed to merely prefixes).
		'''
		if node.is_word:
			words.append(node.text)

		for letter in node.children:
			self.__child_words_for(node.children[letter], words)



	def size(self):
		# By default, get the size of the whole trie, starting at the root
		if not current:
			current = self.root

		count = 1
		for letter in current.children:
			count += self.size(current.children[letter])
		
		return count
	

class TrieNodeGPT:
	def __init__(self):
		self.children = {}
		self.document_ids = set()


class TrieGPT:
	def __init__(self):
		# Initialize a root node.
		self.root = TrieNodeGPT()
	

	def insert(self, word, document_id):
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

		# Pointer is expected to be at the bottom most child given the
		# word. Add the document ID to the set in that node.
		node.document_ids.add(document_id)  # Original
		# node.document_ids.update(document_id)
	

	def search(self, word):
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
	

# Save trie structure (simplified example, might need custom serialization)
def serialize_trie_node(node: TrieNodeGPT):
    return {
        'children': {char: serialize_trie_node(child) for char, child in node.children.items()},
        'document_ids': list(node.document_ids)
    }


# Load trie structure
def deserialize_trie_node(data: Dict):
    node = TrieNodeGPT()
    node.document_ids = set(data['document_ids'])
    node.children = {char: deserialize_trie_node(child) for char, child in data['children'].items()}
    return node


# def group_words_by_starting_char(words):
#     grouped_words = defaultdict(list)

#     for word in words:
#         if word:  # Check if the word is not empty
#             starting_char = word[0].lower()  # Use lowercase to ignore case differences
#             grouped_words[starting_char].append(word)

#     return grouped_words


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


def index_documents_as_ints(doc_to_word_files: List[str], use_json: bool = False) -> Dict:
	doc_to_int = dict()
	int_value = 0

	for file in tqdm(doc_to_word_files):
		data = load_data_file(file, use_json)
		
		articles = list(data.keys())
		for article in articles:
			doc_to_int[article] = int_value
			int_value += 1

	return doc_to_int


def explore_data() -> None:
	# Load config.
	with open("config.json", "r") as f:
		config = json.load(f)

	# IDF path.
	idf_path = config["preprocessing"]["idf_cache_path"]
	d2w_path = config["preprocessing"]["doc_to_words_path"]
	extension = ".msgpack"

	# Doc2Word files.
	d2w_files = sorted([
		os.path.join(d2w_path, file) 
		for file in os.listdir(d2w_path)
		if file.endswith(extension)
	])

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

	# exit()

	digit_count = 0
	alpha_count = 0
	print(f"Number of total words: {len(words)}")

	starting_chars = string.digits + string.ascii_lowercase + string.ascii_uppercase
	for char in tqdm(starting_chars):
		select_words = [
			word for word in words 
			if word.startswith(char) and len(word) < limit
		]
		if char.isdigit():
			digit_count += len(select_words)
		elif char.isalpha():
			alpha_count += len(select_words)
		else:
			continue


	print(f"Number of digit words: {digit_count}")
	print(f"Number of alpha words: {alpha_count}")

	# exit()


	for char in starting_chars:
		if unicodedata.category(char) == "Cc":
			continue

		print(f"Starting char: {char}")
		select_words = [
			word for word in words 
			if word.startswith(char) and len(word) < limit
		]
		print(f"Number of words: {len(select_words)}")
		print(f"Category: {unicodedata.category(char)}")
		try: 
			print(f"Name: {unicodedata.name(char)}")
		except Exception as e:
			pass
		
		trie_path = "./test_" + str(starting_chars.index(char)) + ".msgpack"
		if os.path.exists(trie_path):
			continue

		select_words = [
			word for word in words 
			if word.startswith(char) and len(word) < limit
		]

		trie = Trie()
		print(f"Build Trie starting with {char}")
		print(f"Number of words to store in trie: {len(select_words)}")
		for word in tqdm(select_words):
			trie.insert(word)

		# Write Trie to file.
		trie_dict = serialize_trie_node(trie.root)
		write_data_file(trie_path, trie_dict, False)

	# Build Trie.
	# single_trie = Trie()
	# for word in words:
	# 	single_trie.insert(word)
	
	# Write Trie to file.
	# trie_dict = serialize_trie_node(single_trie)
	# write_data_file("./test.msgpack", trie_dict, False)

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
	#   trie for each alphanumerical starting character.
	# - RAM overhead is around 35GB for just a single trie so using
	#   multprocessing (even on the server) is not advised without 
	#   sufficient memory resources.

	# Size of trie with just words (no documents list):

	pass


def build_trie(limit: int, char: str, d2w_files: List[str], doc_to_int: Dict) -> TrieGPT:
	# List of all alphanumerics.
	alpha_numerics = string.digits + string.ascii_lowercase + string.ascii_uppercase

	# Initialize the trie for the character group.
	char_trie = TrieGPT()

	# Iterate through the document to word files.
	for file in tqdm(d2w_files):
		# Load the doc to word data.
		doc_to_words = load_data_file(file, False)

		# Iterate through each document in the file.
		for doc in list(doc_to_words.keys()):
			# Load the document word frequencies.
			word_freq = doc_to_words[doc]

			# Retrieve the document's numerical ID from the 
			# document to int dictionary.
			doc_id = doc_to_int[doc]

			# Iterate through each word in the word frequency map.
			for word in list(word_freq.keys()):
				# Isolate the first character of the word.
				word_char = word[0]

				# if char == "other" and word_char not in alpha_numerics and len(word) <= limit:
				# 	# Insert the word, document_id pair into the 
				# 	# trie if the current target character is 
				# 	# "other" AND the word length is under the 
				# 	# predefined limit.
				# 	char_trie.insert(word, doc_id)
				# elif (word_char != "other" and word_char != char) or len(word) > limit:
				# 	# Skip words that don't match the target
				# 	# character or have lengths above the 
				# 	# predefined limit.
				# 	continue
				# else:
				# 	# Insert the word, document_id pair into the
				# 	# trie if the current target character matches
				# 	# the current word character AND the word 
				# 	# length is under the predefined limit.
				# 	char_trie.insert(word, doc_id)

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

	# Load paths.
	d2w_path = config["preprocessing"]["doc_to_words_path"]
	trie_path = config["preprocessing"]["trie_cache_path"]

	if not os.path.exists(trie_path):
		os.makedirs(trie_path, exist_ok=True)

	# Doc2Word files.
	d2w_files = sorted([
		os.path.join(d2w_path, file) 
		for file in os.listdir(d2w_path)
		if file.endswith(extension)
	])

	doc_to_int_path = os.path.join(trie_path, "doc_to_int" + extension)
	int_to_doc_path = os.path.join(trie_path, "int_to_doc" + extension)

	###################################################################
	# BUILD/LOAD DOCUMENT IDS
	###################################################################

	# Load or initialize map from documents to unique IDs.
	if not os.path.exists(doc_to_int_path) or not os.path.exists(int_to_doc_path):
		print("Indexing all documents to unique numerical IDs...")
		doc_to_int = index_documents_as_ints(d2w_files, args.use_json)
		int_to_doc = {value: key for key, value in doc_to_int.items()}

		# Save to file.
		write_data_file(doc_to_int_path, doc_to_int, args.use_json)
		write_data_file(int_to_doc_path, int_to_doc, args.use_json)
	else:
		print("Loading all document to unique ID mappings...")
		doc_to_int = load_data_file(doc_to_int_path, False)

	assert doc_to_int is not None

	###################################################################
	# BUILD TRIES
	###################################################################

	# Set character limit to eliminate ridiculously long strings that 
	# are probably not actual english words. Should help counter max-
	# recursion limit error too.
	limit = 60 # Limit was determined based on longest word in English language at 45 characters (Google'd it) but I allowed for some extra space.

	# Initialize a list of all alphanumerics.
	alpha_numerics = string.digits + string.ascii_lowercase + string.ascii_uppercase
	print("Creating tries...")

	# Iterate through all alphanumerics and an "other" category. This
	# will server as our target character to build our tries.
	for char in list(alpha_numerics) + ["other"]:
		print(f"Processing words that start with {char} character{'' if char in alpha_numerics else 's'}")
		path = os.path.join(trie_path, char + "_trie" + extension)

		# Skip if path exists.
		if os.path.exists(path):
			continue

		char_trie = build_trie(limit, char, d2w_files, doc_to_int)

		# Save the trie.
		trie_dict = serialize_trie_node(char_trie.root)
		print(f"Saving trie to {path}")
		write_data_file(path, trie_dict, args.use_json)

		# Delete trie and collect garbage.
		del char_trie
		gc.collect()

	# Load the tries.

	# Run a test search on the loaded tries.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()