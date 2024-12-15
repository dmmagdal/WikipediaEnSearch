# precompute_sparse_vectors.py
# Given the output data from preprocess.py in doc_to_words and 
# word_to_docs, precompute the sparse vector representations of TF-IDF
# and BM25.
# Python 3.11
# Windows/MacOS/Linux


import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import multiprocessing as mp
import os
import shutil
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup
from bs4 import NavigableString, Tag
import msgpack
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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


def clear_folder(folder_path: str) -> None:
	'''
	Clear the contents of the given folder.
	@param: folder_path (str), the (valid) folder that should be 
		emptied of its contents.
	@return: returns nothing.
	'''
	# Iterate through each item found in the folder path.
	for item in os.listdir(folder_path):
		# Get the full path of the item.
		item_path = os.path.join(folder_path, item)

		try:
			# Try and delete the item (folder or file).
			if os.path.isfile(item_path) or os.path.islink(item_path):
				os.unlink(item_path)  # Remove files or symlinks
				print(f"Deleted file: {item_path}")
			elif os.path.isdir(item_path):
				shutil.rmtree(item_path)  # Remove directories and their contents
				print(f"Deleted folder: {item_path}")
		except Exception as e:
			print(f"Failed to delete {item_path}. Reason: {e}")


def clear_staging_folders(staging_folders: List[str]) -> None:
	'''
	Clear the contents of all staging folders
	@param: staging_folders (List[str]), the list of all (valid) 
		staging folders that should be emptied of their contents.
	@return: returns nothing.
	'''
	for folder in staging_folders:
		assert os.path.exists(folder)
		clear_folder(folder)


def isolate_invalid_articles(pages: List[str]) -> List[str]:
	'''
	Isolate the invalid articles away from the valid ones.
	@param: pages (List[str]), the list of articles parsed by 
		beautifulsoup that need to be analyzed.
	@return, returns a list containing the SHA1 strings of all invalid
		articles from the input list.
	'''
	# Initialize a list to store the article SHA1's for all invalid
	# articles.
	redirect_shas = list()

	# Iterate through each article.
	for page_str in tqdm(pages):
		# Parse page with beautifulsoup.
		page = BeautifulSoup(page_str, "lxml")

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
			redirect_shas.append(article_sha1)

	# Return the list of invalid article SHAs.
	return redirect_shas


def get_document_lengths(doc_to_words: Dict[str, Dict[str, int]]) -> List[int]:
	'''
	Given the list of documents, retrieve the lengths of each one.
	@param: doc_to_word (Dict[str, Dict[str, int]]), the map of all
		documents in a file and their respective word frequency 
		mappings as well.
	@return: returns a list containing the document lengths (List[int]).
	'''
	# Initialize a list containing the document lengths.
	document_lengths = list()

	# Iterate through the documents.
	for article in tqdm(doc_to_words.keys()):
		# Compute the document length by taking the sum of all word
		# frequencies in the document.
		doc_length = sum(
			[value for value in doc_to_words[article].values()]
		)

		# Addthe document length to the return list.
		document_lengths.append(doc_length)

	# Return the list of document lengths.
	return document_lengths


def compute_idf(
	word_to_doc_files: List[str], corpus_size: int, use_json: bool = False
) -> None:
	'''
	Compute the inverse document frequency for every word in the 
		corpus.
	@param: word_to_doc_files (List[str]), the list of paths for
		all the word to document mapping files for each text file.
	@param: corpus_size (int), the number of documents that exist 
		in the corpus.
	@param: use_json (bool), whether to read the data file from a 
		JSON or msgpack (default is False).
	@return, returns a dictionary mapping every word to its respective
		inverse document frequency.
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

	# Return the IDF dictionary.
	return word_idf


def compute_sparse_vectors(
	doc_to_words: Dict[str, Dict[str, int]], 
	idf_df: pd.DataFrame, 
	k1: float,
	b: float,
	avg_doc_len: float,
) -> List[Tuple[str, str, int, float, float]]:
	
	# Initialize list to store all tuple data.
	vector_data = list()

	for doc, word_freq in tqdm(doc_to_words.items()):
		doc_len = sum([value for value in word_freq.values()])

		for word, tf in word_freq.items():
			# Compute the TF-IDF.
			tf_idf = tf * idf_df[word]

			# Compute the BM25.
			numerator = idf_df[word] * tf * (k1 + 1)
			denominator = tf + k1 * (
				1 - b + b * (doc_len / avg_doc_len)
			)
			bm25 = numerator / denominator

			# Update return list with data.
			vector_data.append(
				(doc, word, tf, tf_idf, bm25)
			)

	# Return list of tuples.
	return vector_data


def main():
	# Initialize argument parser.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--use_json",
		action="store_true",
		help=""
	)
	parser.add_argument(
		"--num_proc",
		type=int,
		default=1,
		help=""
	)
	parser.add_argument(
		"--num_thread",
		type=int,
		default=1,
		help=""
	)
	parser.add_argument(
		"--clear-staging",
		choices=["before", "after", "both", "neither"],
        default="neither",
        help="Specify when to clean up staging: 'before' processing, 'after' processing, 'both' (before and after processing), or 'neither' (default)."
	) # before, after, both, neither

	# Parse arguments.
	args = parser.parse_args()
	use_json = args.use_json
	extension = ".json" if use_json else ".msgpack"
	num_proc = args.num_proc
	num_thread = args.num_thread
	clear_staging = args.clear_staging

	num_cpus = min(mp.cpu_count(), num_proc)
	max_workers = num_cpus if num_proc > 1 else num_thread

	# Load config file and isolate key variables.
	if not os.path.exists("config.json"):
		print("Could not detect required file config.json in current path.")
		print("Exiting program.")
		exit(1)

	with open("config.json", "r") as f:
		config = json.load(f)

	# Isolate paths.
	preprocessing_paths = config["preprocessing"]

	# Output and staging folder paths.
	idf_staging = preprocessing_paths["staging_idf_path"]
	corpus_staging = preprocessing_paths["staging_corpus_path"]
	redirect_staging = preprocessing_paths["staging_redirect_path"]
	output_folder = preprocessing_paths["sparse_vector_path"]

	# Doc to word and word to doc folder paths.
	doc_to_words_folder = preprocessing_paths["doc_to_words_path"]
	word_to_docs_folder = preprocessing_paths["word_to_docs_path"]

	# Validate word to doc and doc to word folder paths exist.
	d2w_folder_exists = os.path.exists(doc_to_words_folder)
	w2d_folder_exists = os.path.exists(word_to_docs_folder)
	if not d2w_folder_exists or not w2d_folder_exists:
		print(f"Error: Could not find path {doc_to_words_folder} or {word_to_docs_folder}")
		print("Make sure to run preprocess.py before this script.")
		exit(1)

	# Validate word to doc and doc to word folder paths are populated.
	doc_to_words_files = [
		os.path.join(doc_to_words_folder, file)
		for file in os.listdir(doc_to_words_folder)
		if file.endswith(extension)
	]
	word_to_docs_files = [
		os.path.join(word_to_docs_folder, file)
		for file in os.listdir(word_to_docs_folder)
		if file.endswith(extension)
	]
	num_w2d_items = len(doc_to_words_files)
	num_d2w_items = len(word_to_docs_files)
	if num_w2d_items == 0 or num_d2w_items == 0:
		print(f"Error: Path {doc_to_words_folder} or {word_to_docs_folder} were found to be empty")
		print("Make sure to run preprocess.py before this script.")
		exit(1)

	# Create the output folders if necessary.
	if not os.path.exists(idf_staging):
		os.makedirs(idf_staging, exist_ok=True)

	if not os.path.exists(corpus_staging):
		os.makedirs(corpus_staging, exist_ok=True)

	if not os.path.exists(redirect_staging):
		os.makedirs(redirect_staging, exist_ok=True)
		
	if not os.path.exists(output_folder):
		os.makedirs(output_folder, exist_ok=True)

	# Clear staging (if applicable).
	if clear_staging in ["before", "both"]:
		clear_staging_folders(
			[idf_staging, corpus_staging, corpus_staging]
		)

	# XML files containing the actual documents.
	data_folder = "./WikipediaEnDownload/WikipediaData"
	data_xml_files = [
		os.path.join(data_folder, file)
		for file in os.listdir(data_folder)
		if file.endswith(".xml")
	]

	###################################################################
	# Stage 1: Identify Non-Article Documents
	###################################################################
	print("Identifying all non-article documents...")

	# Path to output parquet.
	redirect_path = os.path.join(redirect_staging, "redirects.parquet")

	# Perform processing if the end parquet file is not available.
	if not os.path.exists(redirect_path):
		# Initialize a dictionary to map each file to the SHA1s of 
		# every invalid article in the file.
		redirect_files = dict()

		# Iterate through each file.
		for idx, file in enumerate(data_xml_files):
			# Isolate the basename and print out the current file and
			# its position.
			basename = os.path.basename(file)
			print(f"Processing {basename} ({idx + 1}/{len(data_xml_files)})")

			# Open the file.# Load in file.
			with open(file, "r") as f:
				raw_data = f.read()

			# Parse file with beautifulsoup. Isolate the articles.
			soup = BeautifulSoup(raw_data, "lxml")
			pages = soup.find_all("page")
			if pages is None:
				continue
			else:
				pages = [str(page) for page in pages]

			# Chunk the data to enable concurrency/parallelism.
			chunk_size = math.ceil(len(pages) / max_workers)
			pages_list = [
				pages[i:i + chunk_size] 
				for i in range(0, len(pages), chunk_size)
			]
			args_list = [
				(pages_sublist,) for pages_sublist in pages_list
			]
			sha_list = list()

			# Scan for invalid articles.
			if num_proc > 1:
				with mp.Pool(max_workers) as pool:
					results = pool.starmap(
						isolate_invalid_articles, args_list
					)
					for result in results:
						sha_list += result
			else:
				with ThreadPoolExecutor(max_workers) as executor:
					results = executor.map(
						lambda args: isolate_invalid_articles(*args),
						args_list
					)
					for result in results:
						sha_list += result

			# Update redirect files map with the findings from the current 
			# file.
			redirect_files.update({file: sha_list})

		# Flatten data.
		redirect_data = list()
		for file, invalid_article_shas in redirect_files.items():
			redirect_data.append((file, invalid_article_shas))

		# Convert to PyArrow Table. Table columns:
		# file (str), articles (List[str])
		table = pa.Table.from_pydict({
			"file": [record[0] for record in redirect_data],
			"articles": [record[1] for record in redirect_data],
		})

		# Save to Parquet file (store in staging).
		pq.write_table(table, redirect_path)

	print("All non-article documents have been identified.")
	print(f"Results stored to {redirect_path}") 
	gc.collect()

	###################################################################
	# Stage 2: Corpus Statistics mpute Average Document Length
	###################################################################
	print("Computing corpus level statistics...")

	# Target corpus statistics:
	# corpus size (number of documents)
	# average document length (for BM25)

	# Path to output JSON.
	corpus_path = os.path.join(corpus_staging, "corpus_stats.json")

	# Perform processing if the end json file is not available.
	if not os.path.exists(corpus_path):
		# Load the redirect data and initialize a list to contain all
		# document lengths.
		redirect_files_df = pd.read_parquet(redirect_path)
		document_lengths = list()

		# Iterate through each file.
		for idx, file in enumerate(doc_to_words_files):
			# Isolate the basename and print out the current file and
			# its position.
			basename = os.path.basename(file)
			print(f"Processing {basename} ({idx + 1}/{len(data_xml_files)})")

			# Load the doc to words map.
			doc_to_words = load_data_file(file, use_json)

			# Remove the invalid articles from the keys (effectively
			# ignore invalid articles).
			query_file = os.path.join(
				data_folder,
				basename.replace(extension, ".xml")
			)
			invalid_articles = redirect_files_df.loc[
				redirect_files_df["file"] == query_file, "articles"
			].iloc[0]
			valid_articles = list(
				set(doc_to_words.keys())\
					.difference(invalid_articles)
			)

			# Chunk the data to enable concurrency/parallelism.
			chunk_size = math.ceil(len(valid_articles) / max_workers)
			articles_list = [
				valid_articles[i:i + chunk_size] 
				for i in range(0, len(valid_articles), chunk_size)
			]
			doc_to_words_list = [
				{
					article: doc_to_words[article] 
					for article in article_sublist
				}
				for article_sublist in articles_list
			]
			args_list = [
				# (articles_sublist) 
				# for articles_sublist in articles_list
				(doc_to_words_subdict,)
				for doc_to_words_subdict in doc_to_words_list
			]
			
			# Get every (valid) document's length.
			if num_proc > 1:
				with mp.Pool(max_workers) as pool:
					results = pool.starmap(
						get_document_lengths, args_list
					)
					for result in results:
						document_lengths += result
			else:
				with ThreadPoolExecutor(max_workers) as executor:
					results = executor.map(
						lambda args: get_document_lengths(*args),
						args_list
					)
					for result in results:
						document_lengths += result
		
		# Compute the average document length and the corpus size.
		corpus_size = len(document_lengths)
		avg_doc_len = sum(document_lengths) / corpus_size

		# Store in staging.
		with open(corpus_path, "w+") as f:
			json.dump(
				{
					"corpus_size": corpus_size, 
					"avg_doc_len": avg_doc_len
				}, 
				f,
				indent=4
			)
	
	print("All corpus level statistics have been calculated.")
	print(f"Results stored to {corpus_path}") 
	gc.collect()

	###################################################################
	# Stage 3: Compute Inverse Document Frequency (IDF)
	###################################################################
	print("Precomputing Inverse Document Frequencies and storing to staging...")

	# Path to output parquet.
	idf_path = os.path.join(idf_staging, "idf.parquet")

	# Perform processing if the end parquet file is not available.
	if not os.path.exists(idf_path):
		idf = compute_idf(word_to_docs_files, corpus_size, use_json)

		# Flatten data.
		idf_data = list()
		for word, idf_value in idf.items():
			idf_data.append((word, idf_value))

		# Convert to PyArrow Table. Table columns:
		# word (str),idf (float)
		table = pa.Table.from_pydict({
			"word": [record[0] for record in idf_data],
			"idf": [record[1] for record in idf_data],
		})

		# Save to Parquet file (store in staging).
		pq.write_table(table, idf_path)

	# Store in staging.
	print("All Inverse Document Frequencies have been computed.")
	print(f"Results stored to {idf_path}") 
	gc.collect()

	###################################################################
	# Stage 4: Compute TF-IDF and BM25
	###################################################################
	print("Precomputing TF-IDF and BM25 values...")

	# Load the necessary data for filtering articles and computing the
	# TF-IDF and BM25 values.
	redirect_files_df = pd.read_parquet(redirect_path)
	idf_df = pd.read_parquet(idf_path)
	with open(corpus_path, "r") as f:
		avg_doc_len = json.load(f)["avg_doc_len"]
	k1 = config["bm25_config"]["k1"]
	b = config["bm25_config"]["b"]

	# Iterate through each file.
	for idx, file in enumerate(doc_to_words_files):
		# Isolate the basename and print out the current file and
		# its position.
		basename = os.path.basename(file)
		print(f"Processing {basename} ({idx + 1}/{len(data_xml_files)})")

		# Path to output parquet.
		output_file = os.path.join(
			output_folder, 
			basename.replace(extension, ".parquet")
		)

		# Skipif the output parquet exists.
		if os.path.exists:
			continue

		# Initialize a list object to hold the flattened data 
		# output.
		vector_data = list()

		# Load the doc to words map.
		doc_to_words = load_data_file(file, use_json)

		# Remove the invalid articles from the keys (effectively ignore
		# invalid articles).
		query_file = os.path.join(
			data_folder,
			basename.replace(extension, ".xml")
		)
		invalid_articles = redirect_files_df.loc[
			redirect_files_df["file"] == query_file, "articles"
		].iloc[0]
		valid_articles = list(
			set(doc_to_words.keys())\
				.difference(invalid_articles)
		)

		# Chunk the data to enable concurrency/parallelism.
		chunk_size = math.ceil(len(valid_articles) / max_workers)
		articles_list = [
			valid_articles[i:i + chunk_size] 
			for i in range(0, len(valid_articles), chunk_size)
		]
		doc_to_words_list = [
			{
				article: doc_to_words[article] 
				for article in article_sublist
			}
			for article_sublist in articles_list
		]
		args_list = [
			(doc_to_words_subdict, idf_df, k1, b, avg_doc_len)
			for doc_to_words_subdict in doc_to_words_list
		]
			
		# Calculate each document/word's TF-IDF and BM25.
		if num_proc > 1:
			with mp.Pool(max_workers) as pool:
				results = pool.starmap(
					get_document_lengths, args_list
				)
				for result in results:
					vector_data += result
		else:
			with ThreadPoolExecutor(max_workers) as executor:
				results = executor.map(
					lambda args: get_document_lengths(*args),
					args_list
				)
				for result in results:
					vector_data += result

		# Convert to PyArrow Table. Table columns:
		# doc (str), word (str), tf (int), tf-idf (float), bm25 (float)
		table = pa.Table.from_pydict({
			"doc": [record[0] for record in vector_data],
			"word": [record[1] for record in vector_data],
			"tf": [record[2] for record in vector_data],
			"tf-idf": [record[3] for record in vector_data],
			"bm25": [record[4] for record in vector_data],
		})

		# Save to Parquet file (store in staging).
		pq.write_table(table, output_file)

	print("All TF-IDF and BM25 values have been computed.")
	print(f"Results stored to {output_folder}") 
	gc.collect()

	# Clear staging (if applicable).
	if clear_staging in ["after", "both"]:
		clear_staging_folders(
			[idf_staging, corpus_staging, corpus_staging]
		)

	# Exit the program.
	exit(0)
	

if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()