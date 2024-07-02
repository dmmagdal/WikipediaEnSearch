# convert_w2d.py
# Convert data in word_to_doc JSON file to store count directly instead
# of the list of documents.
# Source: https://stackoverflow.com/questions/43442194/how-do-i-read-and-write-with-msgpack
# Source: https://msgpack.org/index.html
# Source: https://github.com/msgpack/msgpack-python
# Python 3.9
# Windows/MacOS/Linux


import argparse
import hashlib
import json
import os

import msgpack
from tqdm import tqdm

from preprocess import get_datastruct_size


def hashSum(data: str) -> str:
	"""
	Compute the SHA256SUM of the xml data. This is used as part of the
		naming scheme down the road.
	@param: data (str), the raw string data from the xml data.
	@return: returns the SHA256SUM hash.
	"""

	# Initialize the SHA256 hash object.
	sha256 = hashlib.sha256()

	# Update the hash object with the (xml) data.
	sha256.update(data.encode('utf-8'))

	# Return the digested hash object (string).
	return sha256.hexdigest()


def test():
	"""
	Test out the necessary steps to first convert the JSON data from 
		str: List[str] to str: int for the word to doc metadata.
	@param: takes no arguments.
	@return: returns nothing.
	"""
	# Target and source files.
	source_file = "pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d.json"
	target1_file = "pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d_count.json"
	target2_file = "pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d_count.msgpack"

	# NOTE:
	# File sizes (du -sh).
	# 3.5G    pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d.json
	# 25M     pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d_count.json
	# 13M     pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d_count.msgpack
	
	# Load data to dict object.
	with open(source_file, "r") as f:
		contents = json.load(f)

	# Convert data from str: List[str] to str: int.
	for key, value in tqdm(contents.items()):
		assert isinstance(value, list)
		contents[key] = len(value)

	# Save the converted data to JSON.
	with open(target1_file, "w+") as out_f1:
		json.dump(contents, out_f1, indent=4)
		
	# Convert the dict object to string. Be sure to sort keys to 
	# maintain consistency.
	contents_str = json.dumps(contents, sort_keys=True)

	# Hash the data.
	original_hash = hashSum(contents_str)

	# Save the object in a messagepack file.
	with open(target2_file, "wb+") as out_f2:
		packed = msgpack.packb(contents)
		out_f2.write(packed)

	# Load the object from the messagepack file.
	with open(target2_file, "rb") as in_f:
		byte_data = in_f.read()

	# Unpack/deserialize the byte data read from the messagepack file.
	loaded_contents = msgpack.unpackb(byte_data)

	# Convert the (loaded) dict object to string. Be sure to sort keys 
	# to maintain consistency.
	loaded_contents_str = json.dumps(loaded_contents, sort_keys=True)

	# Hash the (loaded) data.
	loaded_hash = hashSum(loaded_contents_str)

	# Compute size of each object.
	print("Data size:")
	original_size = get_datastruct_size(contents)
	print("Loaded data size:")
	load_size = get_datastruct_size(loaded_contents)
	
	# Check to see if the loaded data from message pack matches the
	# original data from the JSON.
	print(f"Object match: {contents == loaded_contents}")
	print(f"Hash match: {original_hash == loaded_hash}")
	print(f"Size match: {original_size == load_size}")


def convert():
	"""
	Convert the existing metadata (JSON data) str: List[str] to 
		str: int for the word to doc metadata.
	@param: takes no arguments.
	@return: returns nothing.
	"""
	# Initialize/load tracking information for files already converted.
	tracking_file = "tracking.txt"
	current_progress = []
	if not os.path.exists(tracking_file):
		open(tracking_file, "w+").close()
	else:
		with open(tracking_file, "r") as f:
			current_progress = f.readlines()

		current_progress = [
			file.rstrip("\n") for file in current_progress
		]

	# Isolate path to word_to_docs metadata folder and verify that the
	# path exists.
	doc_to_words_path = "./metadata/bag_of_words/word_to_docs"

	if not os.path.exists(doc_to_words_path):
		print("Path to doc_to_words JSON file does not exist")
		return

	# Retrieve the list of sorted documents in the metadata folder.
	json_files = sorted([
		os.path.join(doc_to_words_path, file) 
		for file in os.listdir(doc_to_words_path) 
		if file.endswith(".json")
	])

	# Iterate through each file.
	for file in tqdm(json_files):
		# Skip files already processed.
		if file in current_progress:
			continue

		# Load the original data.
		with open(file, "r") as w2d_fr:
			data = json.load(w2d_fr)

		# Iterate through the data overriding the values (List[str])
		# with their new value.
		for key, value in data.items():
			assert isinstance(value, list)
			data[key] = len(value)

		# Write the data back to the JSON file.
		with open(file, "w") as w2d_fw:
			json.dump(data, w2d_fw, indent=4)

		# Convert the JSON data to string and hash it.
		data_str = json.dumps(data, sort_keys=True)
		data_hash = hashSum(data_str)

		# Save the data to msgpack.
		msgpack_file = file.replace("json", "msgpack")
		with open(msgpack_file, "wb+") as out_f:
			packed = msgpack.packb(data)
			out_f.write(packed)

		# Load and verify the data.
		with open(msgpack_file, "rb") as in_f:
			byte_data = in_f.read()

		# Unpack/deserialize the byte data read from the messagepack file.
		loaded_data = msgpack.unpackb(byte_data)

		# Convert the (loaded) dict object to string. Be sure to sort keys 
		# to maintain consistency.
		loaded_data_str = json.dumps(loaded_data, sort_keys=True)

		# Hash the (loaded) data.
		loaded_data_hash = hashSum(loaded_data_str)

		# Confirm the data matches.
		assert data == loaded_data
		assert data_hash == loaded_data_hash

		# Update tracking file.
		current_progress.append(file)
		with open("tracking.txt", "w+") as f:
			f.write("\n".join(current_progress))

	return


def main():
	# Initialize argument parser.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--convert",
		action="store_true",
		help="Specify whether to restart the preprocessing from scratch. Default is false/not specified."
	)
	args = parser.parse_args()

	# Determine which action to do based on the arguments passed in:
	# either convert all existing metadata from the current format to
	# the new one (str: List[str] -> str: int) or run a test on a 
	# select file.
	if args.convert:
		convert()
	else:
		test()

	# Exit the program.
	exit(0)
	

if __name__ == '__main__':
	main()