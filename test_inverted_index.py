# test_inverted_index.py
# Test the InvertedIndex class from search.py.
# Python 3.11
# Windows/MacOS/Linux


import argparse
import gc
import multiprocessing as mp
import os
import random
import time
from typing import List, Tuple

from search import InvertedIndex, load_data_file


def test_index(inverted_index: InvertedIndex, samples: List[Tuple[str, int]], num_workers: int = 1) -> None:
	# Iterate through each sample in the sample group.
	for sample in samples:
		# Identify the top N most common words.
		n = 10
		top_pairs = sorted(sample, key=lambda x: x[1], reverse=True)[:n]

		# Print top N most common words and their document count.
		print(f"Top {n} most common terms:")
		for term, frequency in top_pairs:
			print(f"{term}: {frequency} documents")

		# Isolate only the words from the sample.
		words = [pair[0] for pair in sample]

		# Search the inverted index and print the performance times as
		# well as the number of documents returned.
		start = time.perf_counter()
		results = inverted_index.query(words, num_workers)
		end = time.perf_counter()
		elapsed_ms = round((end - start) * 1_000)
		print(f"Number of documents returned: {len(results)}")
		print(f"Query time: {elapsed_ms} ms")
		print()
	
	print("-" * 32)


def main():
	# Initialize argument parser and parse args.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--use_json",
		action="store_true",
		help="Whether to read from JSON or msgpack files. Default is false/not specified."
	)
	parser.add_argument(
		"--num_proc",
		type=int,
		default=1,
		help="How many processors to use. Default is 1."
	)
	parser.add_argument(
		"--num_thread",
		type=int,
		default=1,
		help="How many threads to use. Default is 1."
	)
	args = parser.parse_args()
	use_json = args.use_json
	extension = ".json" if use_json else ".msgpack"
	num_proc = args.num_proc
	num_thread = args.num_thread

	# Load inverted index files.
	inverted_index_folder = "./metadata/bag_of_words/tries"

	# Initialize class.
	inverted_index = InvertedIndex(
		inverted_index_folder, use_json, args.num_proc > 1
	)

	# Load vocabulary.
	vocab_folder = "./metadata/bag_of_words/word_to_docs"
	vocab_files = [
		os.path.join(vocab_folder, file) 
		for file in os.listdir(vocab_folder)
		if file.endswith(extension)
	]
	vocab = dict()
	for file in vocab_files:
		vocab.update(load_data_file(file, use_json))
	
	vocab_terms = list(vocab.keys())

	# Set random seed and randomly sample from the vocabulary.
	random.seed(1234)
	small_size = 10
	medium_size = 50
	large_size = 150

	# Generate N number of samples per each sample size.
	num_samples = 3
	small_samples = list()
	medium_samples = list()
	large_samples = list()
	for _ in range(num_samples):
		small_samples.append(
			[
				(word, vocab[word])
				for word in random.sample(vocab_terms, small_size)
			]
		)
		medium_samples.append(
			[
				(word, vocab[word])
				for word in random.sample(vocab_terms, medium_size)
			]
		)
		large_samples.append(
			[
				(word, vocab[word])
				for word in random.sample(vocab_terms, large_size)
			]
		)

	# Clean up memory.
	del vocab
	del vocab_terms
	gc.collect()

	# num_workers
	# Threads:
	# 4
	# - Small sample: 
	# 8
	# - Small sample:
	# 16
	# - Small sample: 20 minute mean time
	# Processors:
	# 4
	# - Small sample: 4.5 minute mean time (360s or ~6 minutes mean total)
	# - Medium sample: 7.5 minute mean time (550s or ~11 minutes mean total)
	# - Large sample: 15 minute mean time (1000s or ~20 minutes mean total)
	# 8 OOM'ed
	# 16 OOM'ed
	num_workers = min(mp.cpu_count(), num_proc) if num_proc > 1 else num_thread

	# NOTE:
	# Performance seems to excel under multiprocessing but we quickly
	# get OOM with a small number of processors.

	# Test query times based on size of query.
	print(f"Testing small query size ({small_size} words)")
	test_index(inverted_index, small_samples, num_workers)
	print("=" * 72)
	print(f"Testing medium query size ({medium_size} words)")
	test_index(inverted_index, medium_samples, num_workers)
	print("=" * 72)
	print(f"Testing large query size ({large_size} words)")
	test_index(inverted_index, large_samples, num_workers)
	print("=" * 72)

	# Exit the program.
	exit(0)
	

if __name__ == '__main__':
	main()