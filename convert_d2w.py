# convert_d2w.py
# Convert data in doc_to_word JSON file to msgpack.
# Python 3.9
# Windows/MacOS/Linux


import hashlib
import json
import os

import msgpack
from tqdm import tqdm


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


def convert():
	"""
	Convert the existing JSON file to msgpack.
	@param: takes no arguments.
	@return: returns nothing.
	"""

	# Isolate path to word_to_docs metadata folder and verify that the
	# path exists.
	word_to_docs_path = "./metadata/bag_of_words/doc_to_words"

	if not os.path.exists(word_to_docs_path):
		print("Path to doc_to_words JSON file does not exist")
		return

	# Initialize/load tracking information for files already converted.
	current_progress = sorted([
		os.path.join(word_to_docs_path, file) 
		for file in os.listdir(word_to_docs_path)
		if file.endswith(".msgpack")
	])

	# Retrieve the list of sorted documents in the metadata folder.
	json_files = sorted([
		os.path.join(word_to_docs_path, file) 
		for file in os.listdir(word_to_docs_path) 
		if file.endswith(".json")
	])

	# Iterate through each file.
	for file in tqdm(json_files):
		# Skip files already processed.
		if file in current_progress:
			continue

		# Load the original data.
		with open(file, "r") as d2w_fr:
			data = json.load(d2w_fr)

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

	return


def main():
	convert()

	# Exit the program.
	exit(0)
	

if __name__ == '__main__':
	main()