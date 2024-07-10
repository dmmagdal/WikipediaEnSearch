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

import torch

from search import ReRankSearch, TF_IDF, BM25, VectorSearch
from search import print_results


def test() -> None:
	'''
	Test each of the search processes on the wikipedia dataset.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	# Input values to search engines
	bow_dir = "./metadata/bag_of_words"
	index_dir = "./temp"
	model = "bert"
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"

	###################################################################
	# INITIALIZE SEARCH ENGINES
	###################################################################
	search_1_init_start = time.perf_counter()
	tf_idf = TF_IDF(bow_dir)
	search_1_init_end = time.perf_counter()
	search_1_init_elapsed = search_1_init_end - search_1_init_start
	print(f"Time to initialize TF-IDF search: {search_1_init_elapsed:.6f} seconds")

	search_2_init_start = time.perf_counter()
	bm25 = BM25(bow_dir)
	search_2_init_end = time.perf_counter()
	search_2_init_elapsed = search_2_init_end - search_2_init_start
	print(f"Time to initialize BM25 search: {search_2_init_elapsed:.6f} seconds")
	
	# search_3_init_start = time.perf_counter()
	# vector_search = VectorSearch()
	# search_3_init_end = time.perf_counter()
	# search_3_init_elapsed = search_3_init_end - search_3_init_start
	# print(f"Time to initialize Vector search: {search_3_init_elapsed:.6f} seconds")

	search_4_init_start = time.perf_counter()
	rerank = ReRankSearch(bow_dir, index_dir, model, device=device)
	search_4_init_end = time.perf_counter()
	search_4_init_elapsed = search_4_init_end - search_4_init_start
	print(f"Time to initialize Rerank search: {search_4_init_elapsed:.6f} seconds")

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
		# Title: Spencer Bailey
		# SHA1: rpgkzbz22eatq2nlei6w8hd9uk3ikdy
		# File: pages-articles-multistream7_cd9a3d4ba65dba361f438f398c2fe819f0d736e915f415eec20c12935f4afaec.xml
		"Bailey graduated from [[Pomfret School]] in Pomfret, Connecticut, in 2004. He received a B.A. in English from [[Dickinson College]] in Carlisle, Pennsylvania, in 2008 and an M.S. in journalism from [[Columbia University Graduate School of Journalism]] in 2010.&lt;ref name=\"LinkedIn\"&gt;{{cite web|title=Spencer Bailey LinkedIn|url=http://www.linkedin.com/in/spencercbailey}}&lt;/ref&gt; He wrote his Dickinson College thesis about [[Philip Larkin]] as a jazz poet.&lt;ref name=\"Slow Words\"&gt;{{cite web|title=Spencer Bailey, editor, New York|url=https://www.slow-words.com/spencer-bailey-editor-new-york/}}&lt;/ref&gt;",
		# Title: Johntá Austin
		# SHA1: f52xbwrs52gy3fyoij9mg5hj50748a6
		# File: pages-articles-multistream7_cd9a3d4ba65dba361f438f398c2fe819f0d736e915f415eec20c12935f4afaec.xml
		"Austin grew up singing in church choirs and wanted to become an actor. He interviewed celebrities including [[Michael Jackson]] and [[Michael Jordan]], among others, and in 1993 he made his television debut on ''[[The Arsenio Hall Show]]''. On the show he said he loved singing and was asked to sing with [[Arsenio Hall]]'s band.&lt;ref&gt;[http://www.6.islandrecords.com/site/artist_bio.php?artist_id=649]{{dead link|date=April 2017|bot=InternetArchiveBot|fix-attempted=yes}}&lt;/ref&gt;",
		# Title: Han Pil-hwa
		# SHA1: 7lcqgwzloqtl5zbtr90xb2ny72kdmth
		# File: pages-articles-multistream17_acefc7d7172af538daa3f2a1a262627d63766a857e73e5be9713fd9993fd5cf1.xml
		"After her career in sports, Han has held various offices in politics and sports administration. She became the chief secretary of the [[Speed Skating Association]] February 1986. She has also been deputy director general of the Guidance Bureau for Winter Sports of the [[National Sports Committee]], head of the Technical Guidance Office for Winter Sports of the National Sports Committee, vice-chairwoman of the [[Korea Ice Skating Association]], and vice-chairwoman of the [[Athletic Technique Union]].&lt;ref name=\"Seoul2002\"&gt;{{cite book|title=North Korea Handbook|url=https://books.google.com/books?id=JIlh9nNeadMC&amp;pg=PA781|year=2002|publisher=[[Yonhap News Agency]]|location=Seoul|isbn=978-0-7656-3523-5|page=781}}&lt;/ref&gt;",
		# Title: Exclusive economic zone of Brazil
		# SHA1: 1d90usflcvyw3xwg00s9zq1tlziyasb
		# File: pages-articles-multistream24_dfbf8bc9fa53740888a319295288b448c9605d611edb36e62096df21f8bde98c.xml
		"The area may be expanded to 4.4&amp;nbsp;million square kilometres in view of the Brazilian claim that was submitted to the [[United Nations Convention on the Law of the Sea|United Nations Commission on the Limits of the Continental Shelf (CLCS)]] in 2004.&lt;ref&gt;[https://www.un.org/Depts/los/clcs_new/submissions_files/bra04/bra_exec_sum.pdf UN Continental Shelf and UNCLOS Article 76: Brazilian Submission]&lt;/ref&gt; It is proposed to extend Brazil's continental shelf to 900 thousand square kilometers of marine soil and subsoil, which the country will be able to explore.&lt;ref&gt;[http://www.senado.gov.br/conleg/artigos/direito/DireitosBrasileirosdeZona.pdf Gonçalves, J. B. – Direitos Brasileiros de Zona Econômica Exclusiva...]&lt;/ref&gt; With the extension, the area will become more contiguous, including the areas of Brazilian archipelagos in the [[Atlantic Ocean|South Atlantic]]. The region with the largest Blue Amazon is the Northeast, due to the existence of several islands that are well spaced from each other in a contiguous marine zone (the island of [[Trindade and Martin Vaz|Trindade]] is too far from the coast for the same to occur).",
		# Title: Propositional calculus
		# SHA1: d96gvht6osqafiomyghyxzahe80ygdx
		# File: pages-articles-multistream1_d7b6e7f73b71fa7ec184da2710574f8eab919b22b9dddc4b005fd840000db8ca.xml
		"Propositional logic is typically studied through a [[formal system]] in which [[well-formed formula|formulas]] of a [[formal language]] are [[interpretation (logic)|interpreted]] to represent [[propositions]]. This formal language is the basis for [[Proof calculus|proof systems]], which allow a conclusion to be derived from premises if, and only if, it is a [[logical consequence]] of them. This section will show how this works by formalizing the {{section link||Example argument}}. The formal language for a propositional calculus will be fully specified in {{section link||Language}}, and an overview of proof systems will be given in {{section link||Proof systems}}.",
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
		action="store_true",
		help="Specify whether to run the search engine tests. Default is false/not specified."
	)
	args = parser.parse_args()

	# Depending on the arguments, either run the search tests or just
	# use the general search function.
	if args.test:
		test()
	else:
		search_loop()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()