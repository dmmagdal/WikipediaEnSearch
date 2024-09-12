# json_converter.py
# Convert JSON data from JSON file to msgpack file format.
# Source: https://stackoverflow.com/questions/43442194/how-do-i-read-and-write-with-msgpack
# Source: https://msgpack.org/index.html
# Source: https://github.com/msgpack/msgpack-python
# Python 3.9
# Windows/MacOS/Linux


import hashlib
import json

import msgpack

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


def main():
	# Target and source files.
	# source_file = "pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b.json"
	source_file = "pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d.json"
	# target_file = "pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b.msgpack"
	target_file = "pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d.msgpack"

	# NOTE:
	# File sizes (du -sh).
	# 443M    pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b.json
	# 177M    pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b.msgpack
	# 3.5G    pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b_w2d.json
	# 3.3G    pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b.msgpack
	
	# Load data to dict object.
	with open(source_file, "r") as f:
		contents = json.load(f)
		
	# Convert the dict object to string. Be sure to sort keys to 
	# maintain consistency.
	contents_str = json.dumps(contents, sort_keys=True)

	# Hash the data.
	original_hash = hashSum(contents_str)

	# Save the object in a messagepack file.
	with open(target_file, "wb+") as out_f:
		packed = msgpack.packb(contents)
		out_f.write(packed)

	# Load the object from the messagepack file.
	with open(target_file, "rb") as in_f:
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

	# TODO: Try to see the overhead & improvements of adding gz/gzip
	# compression to the msgpack format.

	# NOTE:
	# This script is to address the issue that the bag-of-words 
	# metadata is so large in file form (JSON) but not so large when 
	# reported in memory.
	#
	# Apparently the memory usage reporting system that I had developed
	# was not accurate to being with. I was using sys.getsizeof() but
	# the caveat is "Only the memory consumption directly attributed to
	# the object is accounted for, not the memory consumption of 
	# objects it refers to." 
	# (source: https://docs.python.org/3.10/library/sys.html#sys.getsizeof)
	# One way to counter this is to use memray package which is only 
	# avialable for MacOS & Linux.
	# (source: https://github.com/bloomberg/memray)
	# Alternatively, since that component of the preprocessing script 
	# was for debugging & readability, it may just be removed entirely.
	# 
	# There is still the issue of the size of the bag of words mapping 
	# data in comparison to the orginal text data. There are multiple 
	# aspects to this issue.
	# 1) JSON as a file format apparently has a lot of overhead. To 
	# counter this, I've opted to use msgpack as the file format over
	# JSON. This is showing a great decrease in file size (for the 
	# doc to word json above, file size goes from 443 MB with JSON
	# to 177 MB with msgpack). The downside to msgpack is that the data
	# is no longer human readable from the file and has to be read into
	# memory to become human readable again.


	# Exit the program.
	exit(0)
	

if __name__ == '__main__':
	main()