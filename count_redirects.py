# count_redirects.py
# Count and mapt the number of articles that are just redirects.


import argparse
import json
import math
import multiprocessing as mp
import os
from typing import List, Tuple

from bs4 import BeautifulSoup
from tqdm import tqdm

from generate_trie import write_data_file


def merge_mappings(results: List[List]) -> Tuple:
	'''
	Merge the results of processing each article in the file from the 
		multiprocessing pool.
	@param: results (list[list]), the list containing the outputs of
		the processing function for each processor.
	@return: returns a tuple of the same processing outputs now 
		aggregated together.
	'''
	# Initialize aggregate variables.
	aggr_article_count = 0
	aggr_redirect_count = 0
	aggr_redirect_files = []

	# Results mappings shape (num_processors, tuple_len). Iterate
	# through each result and update the aggregate variables.
	for result in results:
		# Unpack the result tuple.
		article_count, redirect_count, redirect_files = result

		# Update the aggregate variables.
		aggr_article_count += article_count
		aggr_redirect_count += redirect_count
		aggr_redirect_files += redirect_files

	# Return the aggregated data.
	return aggr_article_count, aggr_redirect_count, aggr_redirect_files


def multiprocess_count(file: str, pages_str: List[str], num_proc: int = 1):
	'''
	Preprocess the text (in multiple processors).
	@param: file (str), the filepath of the current file being
		processed.
	@param: pages_str (List[str]), the raw xml text that is going to be
		processed.
	@param: num_proc (int), the number of processes to use. Default is 
		1.
	@return: returns counters for the number of articles, number of 
		articles with redirect tags, and the list of articles with
		redirect tags.
	'''
	# Break down the list of pages into chunks.
	chunk_size = math.ceil(len(pages_str) / num_proc)
	chunks = [
		pages_str[i:i + chunk_size] 
		for i in range(0, len(pages_str), chunk_size)
	]

	# Define the arguments list.
	arg_list = [(file, chunk) for chunk in chunks]

	# Distribute the arguments among the pool of processes.
	with mp.Pool(processes=num_proc) as pool:
		# Aggregate the results of processes.
		results = pool.starmap(count_redirects, arg_list)

		# Pass the aggregate results tuple to be merged.
		article_count, redirect_count, redirect_files = merge_mappings(
			results
		)

	# Return the different mappings.
	return article_count, redirect_count, redirect_files


def count_redirects(file: str, pages_str: List[str]) -> Tuple[int]:
	'''
	Preprocess the text (in a single thread/process).
	@param: file (str), the filepath of the current file being
		processed.
	@param: pages_str (List[str]), the raw xml text that is going to be
		processed.
	@return: returns counters for the number of articles, number of 
		articles with redirect tags, and the list of articles with
		redirect tags.
	'''
	# Initialize the counter variables and the list of redirected 
	# files.
	article_counter = 0
	redirect_counter = 0
	redirected_files = list()

	# Pass each page string into beautifulsoup.
	pages = [BeautifulSoup(page, "lxml") for page in pages_str]

	# Add article count to article counter.
	article_counter = len(pages)

	# Identify which articles had a redirect tag.
	for page in tqdm(pages):
		redirect = page.find("redirect")
		if redirect is not None:
			# Isolate the article/page's SHA1.
			sha1_tag = page.find("sha1")

			# Skip articles that don't have a SHA1 (should not be 
			# possible but you never know).
			if sha1_tag is None:
				continue

			# Clean article SHA1 text.
			article_sha1 = sha1_tag.get_text()
			article_sha1 = article_sha1.replace(" ", "")\
				.replace("\n", "")
			
			# Update the list of files with redirects.
			file_hash = file + article_sha1
			redirected_files.append(file_hash)

	# Add redirect count to the redirect counter.
	redirect_counter = len(redirected_files)

	# Return the article count, redirect count, and the list of files
	# redirected.
	return article_counter, redirect_counter, redirected_files


def main():
	'''
	Main method. Process the individual wikipedia articles from their
		xml files to create a list of documents that are simply 
		redirects to other documents. This can be used to cut down on
		the number of articles considered for faster bag of words 
		processing during classical search (TF-IDF/BM25).
	@param: takes no arguments.
	@return: returns nothing.
	'''
	###################################################################
	# PROGRAM ARGUMENTS
	###################################################################
	parser = argparse.ArgumentParser()
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

	# Load the configurations from the config JSON.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Itemize files.
	folder = "./WikipediaEnDownload/WikipediaData"
	files = sorted([
		os.path.join(folder, file) for file in os.listdir(folder)
	])
	output_folder = config["preprocessing"]["redirect_cache_path"]
	os.makedirs(output_folder, exist_ok=True)

	# Initialize article and redirect counters.
	article_counter = 0
	redirect_counter = 0

	# Initialize a list of all pages with redirect tags.
	redirected_files = []

	# Iterate through the articles in each file.
	for idx, file in enumerate(files):
		print(f"Processing file {idx + 1}/{len(files)} {file}...")

		# Load file.
		with open(file, "r") as f:
			data = f.read()

		# Parse data with beautiful soup.
		soup = BeautifulSoup(data, 'lxml')

		# Isolate the articles.
		pages = soup.find_all("page")
		pages_str = [str(page) for page in pages]

		if args.num_proc > 1:
			# Determine the number of CPU cores to use (this will be
			# passed down the the multiprocessing function)
			max_proc = min(mp.cpu_count(), args.num_proc)

			article_count, redirect_count, redirect_files = multiprocess_count(
				file, pages_str, num_proc=max_proc
			)
		else:
			article_count, redirect_count, redirect_files = count_redirects(
				file, pages_str
			)

		# Increment the counter variables and the list of files.
		article_counter += article_count
		redirect_counter += redirect_count
		redirected_files += redirect_files

	# Print results.
	print(f"Total article count: {article_counter}")
	print(f"Total redirect count: {redirect_counter}")

	# NOTE:
	# Results found were 23,820,807 total articles and number of
	# redirect articles detected were 11,547,515.

	# Save the list of files with redirects to file.
	# with open("redirect_files.txt", "w+") as f:
	# 	f.write("\n".join(redirected_files))
	extension = ".json" if args.use_json else ".msgpack"
	chunk_size = 1_000_000
	chunks = [
		redirected_files[i:i + chunk_size] 
		for i in range(0, len(redirected_files), chunk_size)
	]
	for idx, chunk in enumerate(chunks):
		filename = "redirect_files_" + str(idx + 1) + extension
		write_data_file(
			os.path.join(output_folder, filename), 
			chunk, 
			args.use_json
		)

	# TODO:
	# I should go back and insert this code to ignore/skip redirect 
	# articles in the bag-of-words preprocessing done in preprocess.py 
	# and follow that up with redoing the metadata computation from 
	# that output.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()