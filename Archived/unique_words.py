# unique_words.py
# Count the number of unique words give the word-to-documents mappings.
# Python 3.9
# Windows/MacOS/Linux


import argparse
import json
import os

from generate_tfidf_cache import load_data_file, write_data_file, compute_idf
from generate_tfidf_cache import get_number_of_documents


def main():
	# Define arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--use_json",
		action="store_true",
		help="Specify whether to load and write metadata to/from JSON files. Default is false/not specified."
	)
	args = parser.parse_args()

	# Open config file.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Declare which extension to use based on the arguments.
	extension = ".json" if args.use_json else ".msgpack"

	# Isolate the word to documents mappings folder.
	preprocessing = config["preprocessing"]
	w2d_metadata_path = preprocessing["word_to_docs_path"]
	d2w_metadata_path = preprocessing["doc_to_words_path"]
	corpus_size = config["tf-idf_config"]["corpus_size"]

	d2w_files = sorted([
		os.path.join(d2w_metadata_path, file) 
		for file in os.listdir(d2w_metadata_path)
		if file.endswith(extension)
	])

	# Load all word to document files.
	w2d_files = sorted(
		[
			os.path.join(w2d_metadata_path, file)
			for file in os.listdir(w2d_metadata_path)
			if file.endswith(extension)
		]
	)

	# Compute the corpus size if necessary.
	if corpus_size == 0:
		corpus_size = get_number_of_documents(d2w_files, args.use_json)

	# Initialize dictionary storing mapping of each unique word to
	# its respective IDF.
	unique_words = dict()

	# Iterate through the word to document files.
	for idx, file in enumerate(w2d_files):
		print(f"Processing file {idx + 1}/{len(w2d_files)} {file}...")

		# Load the file.
		word_to_docs = load_data_file(file, args.use_json)

		# Isolate the words that are unique to the file.
		# file_unique_words = [
		# 	word for word in list(word_to_docs.keys())
		# 	if word not in list(unique_words.keys())
		# ] # optimization implemented but VERY SLOW
		# file_unique_words = [
		# 	word for word in list(word_to_docs.keys())
		# 	if word not in set(unique_words.keys())
		# ] # optimization implemented but VERY SLOW
		file_unique_words = list(
			set(word_to_docs.keys()).difference(set(unique_words.keys()))
		) # optimization implemented and actually resulted not freezing (unlike the implementations above) - I think this has resulted in the seedup/optimziation I was looking for
		# file_unique_words = list(word_to_docs.keys()) # Unoptimized (not tested for performance)

		# Compute the IDFs for each file unique word.
		word_idfs = compute_idf(
			w2d_files, corpus_size, file_unique_words, args.use_json
		)

		# Update the main word to IDF dictionary.
		unique_words.update(word_idfs)

	# Print the number of written words in the corpus.
	print(f"Number of unique words in the corpus: {len(list(unique_words.keys()))}")

	# Save the unique word IDF mappings to file.
	path = "./unique_word_idf" + extension
	write_data_file(path, unique_words, args.use_json)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()