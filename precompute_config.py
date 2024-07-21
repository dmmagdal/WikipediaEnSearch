# precompute_config.py
# Compute certain values for the TF-IDF and BM25 search engines and 
# saving that information to the config file.
# Python 3.9
# Windows/MacOS/Linux


import json
from search import TF_IDF, BM25


def main():
	# Load in config.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Initialize the search engines with their default values.
	tf_idf_bow_dir = config["tf-idf_config"]["bow_dir"]
	bm25_bow_dir = config["tf-idf_config"]["bow_dir"]

	if tf_idf_bow_dir != bm25_bow_dir:
		warning_string = "WARNING!\n" +\
			"Detected bag-of-words directories for TF-IDF and BM25 " +\
			f"to be in different locations ({tf_idf_bow_dir}, " +\
			f"{bm25_bow_dir}), " + "It is usually more efficient " +\
			"to have these directories the same since both use the " +\
			"same data/metadata. If this is not a mistake, proceed " +\
			"with caution."
		print(warning_string)

	tf_idf = TF_IDF(bow_dir=tf_idf_bow_dir)
	bm25 = BM25(bow_dir=bm25_bow_dir)

	if tf_idf_bow_dir == bm25_bow_dir:
		assert tf_idf.corpus_size == bm25.corpus_size,\
			f"Expected corpus_size in TF-IDF and BM25 to match since they are using the same bag-of-words directory path. Recieved {tf_idf.corpus_size} and {bm25.corpus_size}"

	# Update parts of the config.
	config["tf-idf_config"]["corpus_size"] = tf_idf.corpus_size
	config["bm25_config"]["corpus_size"] = bm25.corpus_size
	config["bm25_config"]["avg_doc_len"] = bm25.avg_corpus_len

	# Save the config file.
	with open("config.json", "w") as f:
		json.dump(config, f, indent=4)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()