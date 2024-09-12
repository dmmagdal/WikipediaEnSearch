# generate_cache_data.py
# Generate all required cache data for the search engine.
# Source: https://www.aleksandrhovhannisyan.com/blog/python-trie-data-structure/
# Python 3.9
# Windows/MacOS/Linux


import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import string
from typing import List, Dict, Any, Set, Tuple

from bs4 import BeautifulSoup
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


def get_number_of_documents(doc_to_word_files: List[str], use_json: bool = False) -> int:
		'''
		Count the number of documents recorded in the corpus.
		@param: doc_to_word_files (List[str]), the list of all 
			document to word frequency paths that will be used to take
			inventory of the number of documents in the corpus, 
		@param: use_json (bool), whether to read the data file from a 
			JSON or msgpack (default is False).
		@return, returns the number of documents in the corpus.
		'''
		# NOTE:
		# This does not take into account filtering articles that are 
		# "redirect" or "category articles". Can make this a TODO item
		# for later.

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


def compute_idf(word_to_doc_files: List[str], idf_metadata_path: str, corpus_size: int, use_json: bool = False) -> None:
	'''
		Count the number of documents recorded in the corpus.
		@param: word_to_doc_files (List[str]), the list of paths for
			all the word to document mapping files for each text file.
		@param: idf_metadata_path (str), the path to where the IDF 
			metadata is stored.
		@param: corpus_size (int), the number of documents that exist 
			in the corpus.
		@param: use_json (bool), whether to read the data file from a 
			JSON or msgpack (default is False).
		@return, returns nothing.
	'''
	# Aggregate the word count across all documents.
	word_count = dict()
	for word_to_doc_file in tqdm(word_to_doc_files, "Aggregating word counts"):
		word_to_docs = load_data_file(word_to_doc_file, use_json=use_json)
		for word in list(word_to_docs.keys()):
			if word not in list(word_count.keys()):
				word_count[word] = word_to_docs[word]
			else:
				word_count[word] += word_to_docs[word]

	# Compute the inverse documemnt frequency for all words.
	word_idf = dict()
	for word in tqdm(list(word_count.keys()), f"Computing IDF"):
		word_idf[word] = math.log(corpus_size / word_count[word])

	# Chunk the IDF dictionary and save it.
	CHUNK_SIZE = 5_000_000
	words = list(word_idf.keys())
	chunks = [
		words[i:i + CHUNK_SIZE]
		for i in range(0, len(words), CHUNK_SIZE)
	]
	os.makedirs(idf_metadata_path, exist_ok=True)
	extension = ".json" if use_json else ".msgpack"

	for idx, chunk in enumerate(chunks):
		basename = f"idf_{idx + 1}{extension}"
		path = os.path.join(idf_metadata_path, basename)
		chunk_shard = {word: word_idf[word] for word in chunk}
		write_data_file(path, chunk_shard, use_json)

	return


def build_document_tries(file: str, doc_to_word_file: str, trie_metadata_path: str, limit: int = 60, use_json: bool = False) -> None:
	'''
	Process the wikipedia articles from the file and create the 
		inverted index tries for that file.
	@param: file (str), the path of the data file (containing wikipedia
		data) that is to be loaded.
	@param: doc_to_word_path (str), the path of the document to word 
		frequency mapings that correspond to the file.
	@param: trie_metadata_path (str), the path for the trie metadata 
		corresponding to the file.
	@param: limit (int), the maximum number of characters/string length
		for any word in the articles contained in the file (default is
		60 characters).
	@param: use_json (bool), whether to read/write to/from a JSON or
		msgpack file (default is False).
	@return: returns nothing.
	'''
	###################################################################
	# Load Wikipedia Text Data
	###################################################################

	# Extension.
	extension = ".json" if use_json else ".msgpack"

	# Open file.
	with open(file, "r") as f:
		raw_text = f.read()

	# Parse out all pages.
	soup = BeautifulSoup(raw_text, "lxml")
	pages = soup.find_all("page")

	# Initialize document to document ID and inverse mapping.
	doc_to_int = dict()
	int_to_doc = dict()

	# Index all documents/articles that are not "redirect" or 
	# "category" documents/articles.
	for page in tqdm(pages, f"Parsing text articles"):
		###############################################################
		# Extract article SHASUM
		###############################################################

		# Isolate the article/page's SHA1.
		sha1_tag = page.find("sha1")

		# Skip articles that don't have a SHA1 (should not be possible 
		# but you never know).
		if sha1_tag is None:
			continue

		# Clean article SHA1 text.
		article_sha1 = sha1_tag.get_text()
		article_sha1 = article_sha1.replace(" ", "").replace("\n", "")

		###############################################################
		# Skip "redirect" articles
		###############################################################

		# Isolate the article/page's redirect tag.
		redirect_tag = page.find("redirect")

		# Skip articles that have a redirect tag (they have no 
		# useful information in them).
		if redirect_tag is not None:
			continue

		###############################################################
		# Skip "category" articles
		###############################################################

		# Extract the article text.
		title_tag = page.find("title")

		# Verify that the title data is in the article (all articles
		# should have a title tag).
		assert title_tag is not None

		# Extract the text from each tag.
		title = title_tag.get_text()
		title = title.replace("\n", "").strip()

		# Skip articles that start with "Category:" in their title 
		# (these entries only contain mappings to other categories as
		# part of Wikipedia's category tree structure).
		if title.startswith("Category:"):
			continue

		###############################################################
		# Article validated - Add to tracking
		###############################################################

		# Compute the file hash.
		file_hash = file + article_sha1
		
		# Compute the document ID for this article and insert the 
		# mappings to their respective dictionaries.
		document_id = len(list(doc_to_int.keys()))
		doc_to_int[file_hash] = document_id
		int_to_doc[str(document_id)] = file_hash

	# Clean up memory.
	del raw_text
	del soup
	del pages
	gc.collect()

	###################################################################
	# Build Inverted Index Tries
	###################################################################

	# Initialize path to trie.
	os.makedirs(trie_metadata_path, exist_ok=True)

	# Save document to document ID and inverse mappings.
	write_data_file(
		os.path.join(trie_metadata_path, f"doc_to_int{extension}"),
		doc_to_int,
		use_json
	)
	write_data_file(
		os.path.join(trie_metadata_path, f"int_to_doc{extension}"),
		int_to_doc,
		use_json
	)

	# Open doc_to_word file and filter out all documents/articles not 
	# in the index.
	doc_to_words = load_data_file(doc_to_word_file, use_json)

	# Group the words via their starting character.
	alpha_numerics = string.digits + string.ascii_lowercase

	# Build and save tries for each starting character for the file.
	for char in tqdm(list(alpha_numerics) + ["other"], f"Building inverted index tries"):
		# Initialize a trie for the starting word character.
		character_trie = TrieGPT()

		for doc in list(doc_to_int.keys()):
			# Isolate words from the document. Filter out words that
			# exceed the defined character/string length limit.
			words = [
				word for word in list(doc_to_words[doc].keys())
				if len(word) <= limit
			]

			# Retrieve the document ID.
			document_id = doc_to_int[doc]

			# Iterate through each word.
			for word in words:
				word_char = word[0]

				# Add word to the trie if the starting character 
				# matches the current character.
				if char == "other" and word_char not in alpha_numerics:
					character_trie.insert(word, document_id)
				elif word_char == char:
					character_trie.insert(word, document_id)
		
		# Save the trie.
		path = os.path.join(
			trie_metadata_path,
			f"{char}_trie_slim{extension}"
		)
		save_trie(character_trie, path, use_json)

		# Clean up memory.
		del character_trie
		gc.collect()

	# Clean up memory.
	del doc_to_words
	del doc_to_int
	del int_to_doc
	gc.collect()
	return


def build_inverted_index(files: List[str], d2w_data_files: List[str], trie_metadata_path: str, limit: int = 60, use_json: bool = False) -> None:
	'''
	Build the inverted index for all specified Wikipedia data files.
	@param: files (List[str]), the paths of the data file (containing 
		wikipedia data) that is to be processed.
	@param: d2w_data_files (List[str]), the paths of the document to 
		word frequency mapings that correspond to the files.
	@param: trie_metadata_path (str), the path for the trie metadata 
		corresponding to the files.
	@param: limit (int), the maximum number of characters/string length
		for any word in the articles contained in the file (default is
		60 characters).
	@param: use_json (bool), whether to read/write to/from a JSON or
		msgpack file (default is False).
	@return: returns nothing.
	'''
	for idx, file in enumerate(files):
		# Isolate the basename of the file and the corresponding
		# doc to words file.
		basename = os.path.basename(file).replace(".xml", "")
		d2w_file = d2w_data_files[idx]
		trie_folder_path = os.path.join(
			trie_metadata_path, basename
		)

		# Build the inverted index tries from the file.
		build_document_tries(
			file, d2w_file, trie_folder_path, 
			limit=limit, use_json=use_json
		)
	
	return


def main():
	'''
	Main method. Process the word to document and document to word 
		metadata from their respective files to create word to IDF and
		file level inverted index for bag of words processing during 
		classical search (TF-IDF, BM25).
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
		"--limit",
		type=int, 
		default=60, 
		help="Max number of characters per word. Default is 60."
	)
	parser.add_argument(
		"--use_json",
		action="store_true",
		help="Specify whether to load and write metadata to/from JSON files. Default is false/not specified."
	)
	parser.add_argument(
		"--idf",
		action="store_true",
		help="Specify whether to precompute and cache the Inverse Document Frequencies (IDF) of every word. Default is false/not specified."
	)
	parser.add_argument(
		"--trie",
		action="store_true",
		help="Specify whether to precompute and cache the inverted index tries for every file in the Wikipedia data. Default is false/not specified."
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
	idf_metadata_path = preprocessing["idf_cache_path"]
	trie_metadata_path = preprocessing["trie_cache_path"]

	# Initialize the cache paths if necessary.
	if not os.path.exists(idf_metadata_path):
		os.makedirs(idf_metadata_path, exist_ok=True)

	# Initialize the IDF cache path if necessary.
	if not os.path.exists(trie_metadata_path):
		os.makedirs(trie_metadata_path, exist_ok=True)

	# Verify metadata directory paths exist.
	if not os.path.exists(d2w_metadata_path):
		print(f"Bag-of-words document-to-words metadata folder not initialized.")
		print(f"Please initialized folder by downloading the metadata from huggingface.")
		exit(1)

	if not os.path.exists(w2d_metadata_path):
		print(f"Bag-of-words word-to-documents metadata folder not initialized.")
		print(f"Please initialized folder by downloading the metadata from huggingface.")
		exit(1)

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
		print(f"Follow the README.md for instructions on how to download the files from huggingface or run the preprocess.py file to generate the files.")
		exit(1)

	if len(w2d_data_files) == 0:
		print(f"Bag-of-words word-to-documents metadata folder has no files.")
		print(f"Follow the README.md for instructions on how to download the files from huggingface or run the preprocess.py file to generate the files.")
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

	# Folder and files containing the actual Wikipedia data.
	article_folder_path = "./WikipediaEnDownload/WikipediaData"

	if not os.path.exists(article_folder_path):
		print(f"Wikipedia downloaded data folder ({article_folder_path}) not initialized.")
		print(f"Please initialized folder by initializing the WikipediaEnDownload submodule and downloading the data from huggingface.")
		exit(1)

	files = [
		os.path.join(
			article_folder_path, 
			os.path.basename(file).replace(extension, ".xml")
		)
		for file in d2w_files
	]

	# NOTE:
	# No need for sorting files list since it should line up exactly 
	# with the doc_to_words and words_to_doc files.

	all_files_available = all([os.path.exists(file) for file in files])
	if len(files) == 0 or not all_files_available:
		print(f"Wikipedia downloaded data folder has no files.")
		print(f"Follow the README.md for instructions on how to download the files from huggingface.")
		exit(1)

	###################################################################
	# INVERSE DOCUMENT FREQUENCY
	###################################################################
	if args.idf:
		# NOTE:
		# No real good way to parallelize with multiprocessing. Should
		# be relatively alright with just using single processor for 
		# this part.

		# Compute the corpus size if needed.
		corpus_size = 0
		corpus_size_1 = config["bm25_config"]["corpus_size"]
		corpus_size_2 = config["tf-idf_config"]["corpus_size"]
		if corpus_size_1 != corpus_size_2 or corpus_size_1 == 0:
			corpus_size = get_number_of_documents(
				d2w_data_files, use_json=args.use_json
			)

			# Update config and save.
			config["bm25_config"]["corpus_size"] = corpus_size
			config["tf-idf_config"]["corpus_size"] = corpus_size
			with open("config.json", "w+") as f:
				json.dump(config, f, indent=4)
		else:
			corpus_size = corpus_size_1

		assert corpus_size != 0, "Corpus size (required for IDF computation) should be non-zero"

		# Compute and save the inverse document frequency of all words
		# in the corpus.
		compute_idf(
			w2d_files, idf_metadata_path, corpus_size, 
			use_json=args.use_json
		)

	###################################################################
	# INVERTED INDEX TRIES
	###################################################################
	if args.trie:
		# Determine the number of processors to use.
		num_proc = min(args.num_proc, mp.cpu_count())

		# Break down the list of pages into chunks.
		chunk_size = math.ceil(len(files) / num_proc)
		chunks = [
			(files[i:i + chunk_size], d2w_data_files[i:i + chunk_size]) 
			for i in range(0, len(files), chunk_size)
		]

		# Define the arguments list.
		arg_list = [
			(
				files_chunk, 
				d2w_data_files_chunk, 
				trie_metadata_path, 
				args.limit, 
				args.use_json
			) 
			for files_chunk, d2w_data_files_chunk in chunks
		]

		# Distribute the arguments among the pool of processes.
		with mp.Pool(processes=num_proc) as pool:
			# Distribute the tasks around the functions (no need to
			# aggregate results because the function returns nothing).
			pool.starmap(build_inverted_index, arg_list)
		
		# NOTE:
		# Single processor on server looks like it will take around 36
		# (to maybe 48) hours to complete. Ran with 8 cores and it took
		# around 5 hours.

	# Exit the program.
	exit(0)
	

if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()