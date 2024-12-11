# msgpack_converter.py
# Convert data from msgpac file to parquet file format.
# Windows/MacOS/Linux


import os
from typing import Dict

import msgpack
from tqdm import tqdm
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


def main():
	# Source files.
	folder1 = "./metadata/bag_of_words/doc_to_words"
	source_files_1 = [
		os.path.join(folder1, file)
		for file in os.listdir(folder1)
		if file.endswith(".msgpack")
	]
	
	print("Converting doc_to_words from msgpack to parquet")
	for file in tqdm(source_files_1):
		# Identify target file.
		target_file = file.replace(".msgpack", ".parquet")
		if os.path.exists(target_file):
			continue

		# Load data from msgpack file.
		msgpack_data = load_data_from_msgpack(file)

		# Flatten data.
		data = []
		for doc, word_freqs in msgpack_data.items():
			for word, freq in word_freqs.items():
				data.append((doc, word, freq))

		# Convert to PyArrow Table.
		table = pa.Table.from_pydict({
			"doc": [record[0] for record in data],
			"word": [record[1] for record in data],
			"freq": [record[2] for record in data]
		})

		# Save to Parquet file.
		pq.write_table(table, target_file)

	folder2 = "./metadata/bag_of_words/word_to_docs"
	source_files_2 = [
		os.path.join(folder2, file)
		for file in os.listdir(folder2)
		if file.endswith(".msgpack")
	]

	print("Converting word_to_docs from msgpack to parquet")
	for file in tqdm(source_files_2):
		# Identify target file.
		target_file = file.replace(".msgpack", ".parquet")
		if os.path.exists(target_file):
			continue

		# Load data from msgpack file.
		msgpack_data = load_data_from_msgpack(file)

		# Flatten data.
		data = []
		for word, doc_count in msgpack_data.items():
			data.append((word, doc_count))

		# Convert to PyArrow Table.
		table = pa.Table.from_pydict({
			"word": [record[0] for record in data],
			"doc_count": [record[1] for record in data]
		})

		# Save to Parquet file.
		pq.write_table(table, target_file)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()