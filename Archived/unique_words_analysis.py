# unique_words_analysis.py
# Conduct a quick analysis on the data produced by unique_words.py.
# This information will be use to research how viable creating a trie
# (prefix tree) is to help filter out documentas/articles that are not
# relevant given the words in the query.
# Python 3.9
# Windows/MacOS/Linux


import json
import math
import string

from generate_cache import load_data_file


def main():
	path = "./unique_word_idf.msgpack"

	# Load the data.
	data = load_data_file(path, False)

	# Compute the length of the longest string.
	longest_str = max([len(word) for word in list(data.keys())])
	print(f"Longest string length: {longest_str}")
	print(f"Log of longest string length: {math.log(longest_str)}")

	# Compute the frequency at which each (printable) character starts
	# a word.
	char_counts = {char: 0 for char in string.printable}
	for word in list(data.keys()):
		for char in string.printable:
			if word[0] == char:
				char_counts[char] += 1

	print(f"Frequency of each character starting a word:")
	print(json.dumps(char_counts, indent=4))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()