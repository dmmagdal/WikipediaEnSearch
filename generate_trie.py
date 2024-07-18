# generate_trie.py
# Generate a trie (prefix tree) to map documents to individual words.
# Source: https://www.aleksandrhovhannisyan.com/blog/python-trie-data-structure/
# Python 3.9
# Windows/MacOS/Linux


import argparse
from collections import defaultdict
import gc
import json
import os
from typing import List, Dict

import msgpack
from tqdm import tqdm


class TrieNode:
	def __init__(self, text="") -> None:
		self.text = text
		self.children = dict()
		self.is_word = False
	

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
				node.children[char] = TrieNode()

			# Set the pointer to the child node.
			node = node.children[char]

		# Pointer is expected to be at the bottom most child given the
		# word. Add the document ID to the set in that node.
		node.document_ids.add(document_id)
	

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


def group_words_by_starting_char(words):
    grouped_words = defaultdict(list)

    for word in words:
        if word:  # Check if the word is not empty
            starting_char = word[0].lower()  # Use lowercase to ignore case differences
            grouped_words[starting_char].append(word)

    return grouped_words


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


def explore_data() -> None:
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

	# Build Trie.
	single_trie = Trie()
	for word in words:
		single_trie.insert(word)
	
	# Write Trie to file.
	trie_dict = serialize_trie_node(single_trie)
	write_data_file("./test.msgpack", trie_dict, False)

	# Size of trie with just words (no documents list):

	pass


def main():
	explore_data()

	# Initialize a dictionary with a tree for every starting character 
	# possible.
	

	# Iterate through all document to word files and insert the unique 
	# (document, word) pair into the respective trie.

	# Save the tries.

	# Load the tries.

	# Run a test search on the loaded tries.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()