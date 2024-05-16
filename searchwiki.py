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


def test() -> None:
	'''
	Test each of the search processes on the wikipedia dataset.
	@param: takes no arguments.
	@return: returns nothing.
	'''

	###################################################################
	# INITIALIZE SEARCH ENGINES
	###################################################################
	rerank = ReRankSearch()
	bm25 = BM25()
	tf_idf = TF_IDF()
	vector_search = VectorSearch()


	###################################################################
	# EXACT PASSAGE RECALL
	###################################################################


	###################################################################
	# GENERAL QUERY
	###################################################################
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