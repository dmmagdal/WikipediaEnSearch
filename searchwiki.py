# searchwiki.py
# Run a search on the downloaded wikipedia data. The wikipediate data
# should already be downloaded, extracted, and preprocessed by the
# WikipediaEnDownload submodule as well as preprocess.py in this repo.
# Python 3.9
# Windows/MacOS/Linux


import argparse
import os
import json
import time
from search import ReRankSearch, TF_IDF, BM25, VectorSearch
from search import print_results


def test() -> None:
	'''
	Test each of the search processes on the wikipedia dataset.
	@param: takes no arguments.
	@return: returns nothing.
	'''

	###################################################################
	# INITIALIZE SEARCH ENGINES
	###################################################################
	search_1_init_start = time.perf_counter()
	tf_idf = TF_IDF()
	search_1_init_end = time.perf_counter()
	search_1_init_elapsed = search_1_init_end - search_1_init_start
	print(f"Time to initialize TF-IDF search: {search_1_init_elapsed:.6f} seconds")

	search_2_init_start = time.perf_counter()
	bm25 = BM25()
	search_2_init_end = time.perf_counter()
	search_2_init_elapsed = search_2_init_end - search_2_init_start
	print(f"Time to initialize BM25 search: {search_2_init_elapsed:.6f} seconds")
	
	# search_3_init_start = time.perf_counter()
	# vector_search = VectorSearch()
	# search_3_init_end = time.perf_counter()
	# search_3_init_elapsed = search_3_init_end - search_3_init_start
	# print(f"Time to initialize Vector search: {search_3_init_elapsed:.6f} seconds")

	search_4_init_start = time.perf_counter()
	rerank = ReRankSearch()
	search_4_init_end = time.perf_counter()
	search_4_init_elapsed = search_4_init_end - search_4_init_start
	print(f"Time to initialize Vector search: {search_4_init_elapsed:.6f} seconds")

	search_engines = [
		("tf-idf", tf_idf), 
		("bm25", bm25), 
		# ("vector", vector_search),
		("rerank", rerank)
	]

	###################################################################
	# EXACT PASSAGE RECALL
	###################################################################
	# Given passages that are directly pulled from random articles, 
	# determine if the passage each search engine retrieves is correct.
	query_passages = [

	]
	print("=" * 72)

	# Iterate through each search engine.
	for name, engine in search_engines:
		# Search engine banner text.
		print(f"Searching with {name}")
		search_times = []

		# Iterate through each passage and run the search with the 
		# search engine.
		for query in query_passages:
			# Run the search and track the time it takes to run the 
			# search.
			query_search_start = time.perf_counter()
			results = engine.search(query)
			query_search_end = time.perf_counter()
			query_search_elapsed = query_search_end - query_search_start

			# Print out the search time and the search results.
			print(f"Search returned in {query_search_elapsed:.6f} seconds")
			print()
			print_results(results, search_type=name)

			# Append the search time to a list.
			search_times.append(query_search_elapsed)

		# Compute and print the average search time.
		avg_search_time = sum(search_times) / len(search_times)
		print(f"Average search time: {avg_search_time:.6f} seconds")
		print("=" * 72)

	###################################################################
	# GENERAL QUERY
	###################################################################
	# Given passages that have some relative connection to random 
	# articles, determine if the passage each search engine retrieves 
	# is correct.
	query_text = [

	]
	
	for name, engine in search_engines:
		pass
	pass


def search_loop() -> None:
	'''
	Run an infinite loop (or until the exit phrase is specified) to
		perform search on wikipedia.
	@param: takes no arguments.
	@return: returns nothing.
	'''

	# Read in the title text (ascii art).
	with open("title.txt", "r") as f:
		title = f.read()

	exit_phrase = "Exit Search"
	print(title)
	print()
	search_query = input("> ")
	pass


def main() -> None:
	'''
	Main method. Will either run search engine tests or interactive
		search depending on the program arguments.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	# Set up argument parser.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--test",
		action="store_false",
		help="Specify whether to run the search engine tests. Default is true/not specified."
	)

	# Depending on the arguments, either run the search tests or just
	# use the general search function.
	if parser.test:
		test()
	else:
		search_loop()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()