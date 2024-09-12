# zip_json.py
# Zip JSON files to be smaller.
# Source: https://medium.com/@busybus/zipjson-3ed15f8ea85d
# Python 3.9
# Windows/MacOS/Linux


import base64
import json
import os
import time
from typing import Dict, List
import zlib
import zipfile

from tqdm import tqdm

from generate_trie import load_data_file


ZIPJSON_KEY = 'base64(zip(o))'


def compress(j):

	j = {
		ZIPJSON_KEY: base64.b64encode(
			zlib.compress(
				json.dumps(j).encode('utf-8')
			)
		).decode('ascii')
	}

	return j
	

def decompress(j, insist=True):
	# json.loads(zlib.decompress(base64.b64decode(_['base64(zip(o))'])))
	try:
		assert (j[ZIPJSON_KEY])
		assert (set(j.keys()) == {ZIPJSON_KEY})
	except:
		if insist:
			raise RuntimeError("JSON not in the expected format {" + str(ZIPJSON_KEY) + ": zipstring}")
		else:
			return j

	try:
		j = zlib.decompress(base64.b64decode(j[ZIPJSON_KEY]))
	except:
		raise RuntimeError("Could not decode/unzip the contents")

	try:
		j = json.loads(j)
	except:
		raise RuntimeError("Could interpret the unzipped contents")

	return j


def write_zip_json(filepath: str, zip_filepath: str, data: Dict | List) -> None:
	with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
		# zipf.write(filepath, os.path.basename(filepath))
		zipf.writestr(filepath, json.dumps(data, indent=4))


def load_zip_json(filepath: str, zip_filepath: str) -> Dict | List:
	with zipfile.ZipFile(zip_filepath, "r") as zipf:
		with zipf.open(filepath, "r") as jsonf:
			json_data = jsonf.read() # assumes UTF-8 decoding
			# return json.load(jsonf)
	
	return json.loads(json_data)


def main():
	# Isolate all the JSON and msgpack files from the doc_to_words 
	# folder.
	bow_folder = "./metadata/bag_of_words/doc_to_words"
	json_files = sorted([
		os.path.join(bow_folder, file)
		for file in os.listdir(bow_folder)
		if file.endswith(".json")
	])
	msgpack_files = sorted([
		os.path.join(bow_folder, file)
		for file in os.listdir(bow_folder)
		if file.endswith(".msgpack")
	])

	# Initialize lists to keep track of the load times for the 
	# respective json-zip and msgpack files.
	zip_load_times = list()
	msgpack_load_times = list()

	# Process the JSON files.
	for idx, file in enumerate(tqdm(json_files)):
		# Read in the data.
		with open(file, "r") as f:
			data = json.load(f)

		# Set the zip file name.
		zip_file = file.replace(".json", ".zip")

		# Write the data to a zipped JSON.
		write_zip_json(file, zip_file, data)

		# Load the data from the zipped JSON.
		zip_start = time.perf_counter()
		loaded_data = load_zip_json(file, zip_file)
		zip_end = time.perf_counter()
		zip_time = zip_end - zip_start
		zip_load_times.append(zip_time)

		# load the data from the msgpack.
		msgpack_start = time.perf_counter()
		load_data_file(
			msgpack_files[idx], False
		)
		msgpack_end = time.perf_counter()
		msgpack_time = msgpack_end - msgpack_start
		msgpack_load_times.append(msgpack_time)

		# Verify the contents did not change.
		assert data == loaded_data

	# NOTE:
	# - It took around 2 hours for this script to run on all the files 
	# in the doc_to_words metadata folder. All verification passed.
	# - On average, msgpack, despite resulting in larger disk space 
	# overhead, is much faster to load compared to json-zip.

	# TODO:
	# - Integrate with all existing functions that save/load data 
	# to/from JSON.

	# Print file reading stats between json-zip and msgpack files.
	print(f"Fastest Load Times:")
	print(f"\tjson-zip: {min(zip_load_times):.6f} seconds")
	print(f"\tmsgpack: {min(msgpack_load_times):.6f} seconds")

	print(f"Slowest Load Times:")
	print(f"\tjson-zip: {max(zip_load_times):.6f} seconds")
	print(f"\tmsgpack: {max(msgpack_load_times):.6f} seconds")

	print(f"Mean/Avg Load Times:")
	print(f"\tjson-zip: {(sum(zip_load_times) / len(zip_load_times)):.6f} seconds")
	print(f"\tmsgpack: {(sum(msgpack_load_times) / len(msgpack_load_times)):.6f} seconds")

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()