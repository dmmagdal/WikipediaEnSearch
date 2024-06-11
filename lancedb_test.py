# lancedb_test.py
# Test core functions of lancedb as a vector DB.
# Python 3.9
# Windows/MacOS/Linux


import copy
import os
import json
import shutil
import sys

from bs4 import BeautifulSoup
import lancedb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline

from preprocess import process_page, load_model, vector_preprocessing


def main():
	# Open files.
	file = "pages-articles-multistream10_288ec473bfbc27d7686cfee3396c7ea4db31c2f7b2b967778e1e3a21c08a583b.xml"
	with open(file, "r") as f:
		contents = f.read()

	with open("config.json", "r") as j:
		config = json.load(j)

	# Parse file contents with beautiful soup.
	soup = BeautifulSoup(contents, "lxml")
	pages = soup.find_all("page")

	# Subsample the articles from 150,000 to 1,000 articles.
	pages = pages[:1_000]
	pages = pages[:100]	# subsample 100 to go to vector DB functionalities faster OR run 1,000 subsample on Nvidia GPU

	# Load tokenizer and model.
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	tokenizer, model = load_model(config, device)
	# pipe = pipeline(
	# 	"feature-extraction", model=model, tokenizer=tokenizer
	# )

	# Test tokenization encoding & decoding with and without padding.
	text = "hello"
	unpadded_tokens = tokenizer.encode(
		text, 
		add_special_tokens=False,
		# return_tensors="pt"
	)
	padded_tokens = tokenizer.encode(
		text, 
		add_special_tokens=False, 
		padding="max_length",
		# return_tensors="pt"
	) # See https://huggingface.co/docs/transformers/en/pad_truncation#padding-and-truncation and https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer.encode
	# print(f"unpadded len {len(unpadded_tokens)}")
	# print(f"padded len {len(padded_tokens)}")
	# print(padded_tokens)
	# print(unpadded_tokens)
	assert len(unpadded_tokens) < len(padded_tokens), "Padded token sequence expected to be longer than unpadded"

	unpadded_text = tokenizer.decode(unpadded_tokens)
	padded_text = tokenizer.decode(
		padded_tokens, 
		skip_special_tokens=True
	) # See https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer.decode
	# print(f"unpadded text ({len(unpadded_text)}): {unpadded_text}")
	# print(f"padded text ({len(padded_text)}): {padded_text}")
	assert unpadded_text == padded_text, "Decoded texts were expected to be the same"

	# NOTE:
	# Current preprocessing (text chunking + feature extractioon/
	# embeddings) on M2 Macbook Pro is around 26 minutes for the slice
	# of pages that are currently in use. The script is using Pytorch's
	# MPS backend for Apple Silicon for the sake of faster embedding.

	# Initialize list to contain all the vector metadata for the texts/
	# articles parsed.
	vector_metadata = []

	# Iterate through each page and apply the processing for vector
	# data.
	for page in tqdm(pages):
		# Article SHA1 hash.
		sha = page.find("sha1")
		if sha is None:
			continue

		sha = sha.get_text()
		# print(f"SHA: {sha}")
		# print(f"{page.find('title').get_text()}")

		# Load the text from the page.
		text = process_page(page)

		# Tokenize the text and split the tokens into chunks
		# appropriately sized for the model.
		chunk_metadata = vector_preprocessing(text, config, tokenizer)
		# print(json.dumps(chunk_metadata, indent=4))

		# Complete the rest of the preprocessing (embed the text
		# chunks) and update the metadata.
		for idx, chunk in enumerate(chunk_metadata):
			# Update the chunk metadata with the file and article SHA1.
			chunk.update({"SHA1": sha, "file": file})

			# Get original text.
			text_idx = chunk["text_idx"]
			text_len = chunk["text_len"]
			text_chunk = text[text_idx: text_idx + text_len]
			# print(text_chunk)
			
			# Pass text to feature extraction pipeline.
			# output = pipe(
			# 	text_chunk, 
			# 	add_special_tokens=False, 
			# 	padding="max_length", 
			# 	return_tensors="pt"
			# )
			# Returns a list of only 1 tensor of shape (seq_len,
			# emb_dim) where seq_len is the tokenized sequence length
			# unpadded (despite trying to specify padding).
			# print(output[0].shape)

			# Pass text to tokenizer and then to model. Make sure to
			# disable gradients and pass the tokenized text to the
			# appropriate (hardware) device.
			with torch.no_grad():
				output = model(
					**tokenizer(
						text_chunk, 
						add_special_tokens=False, 
						padding="max_length", 
						return_tensors="pt"
					).to(device)
				)
				# Returns a list of only 2 tensors. The first tensor is
				# the last hidden state with shape (batch_size,
				# seq_len, emb_dim) where seq_len is the tokenized
				# sequence length padded (as specified above in the
				# call to the tokenizer above). The second tensor is
				# the pooled output with shape (batch_size, emb_dim).
				# print(output)
				# print(output[0].shape)
				# print(output[1].shape)

				# Take the average of the embedding across the seq_len
				# dimension to get the embedding (results in the tensor
				# of shape (batch_size, emb_dim)). Note that this is
				# NOT the equivalent to the pooled output tensor.
				embedding = output[0].mean(dim=1)

				# Apply the following transformations to allow the
				# embedding to be compatible with being stored in the
				# vector DB (lancedb):
				#	1) Send the embedding to CPU (if it's not already 
				#		there)
				#	2) Convert the embedding to numpy and flatten the
				#		embedding to a 1D array
				embedding = embedding.to("cpu")
				embedding = embedding.numpy()[0]
				# print(embedding.shape)
				# print(torch.equal(embedding, output[1]))
				# exit()
			
			# Update the chunk metadata with the embedding.
			# chunk.update({"embedding": embedding})
			chunk.update({"vector": embedding})	# lancedb requires the embedding data be under the "vector" name

			# Update the metadata with the new chunk.
			chunk_metadata[idx] = chunk

		# Add the metadata to the list.
		vector_metadata += chunk_metadata
			
	# Get an idea of the size of the metadata.
	size_in_bytes = sys.getsizeof(vector_metadata)
	GB_SIZE = 1024 * 1024 * 1024
	MB_SIZE = 1024 * 1024
	KB_SIZE = 1024
	size_in_gb = size_in_bytes / GB_SIZE
	size_in_mb = size_in_bytes / MB_SIZE
	size_in_kb = size_in_bytes / KB_SIZE
	if size_in_gb > 1:
		print(f"Vector metadata has reached over 1GB in size ({round(size_in_gb, 2)} GB)")
	elif size_in_mb > 1: 
		print(f"Vector metadata has reached over 1MB in size ({round(size_in_mb, 2)} MB)")
	elif size_in_kb > 1: 
		print(f"Vector metadata has reached over 1KB in size ({round(size_in_kb, 2)} KB)")
	print(f"Size of vector metadata (in bytes): {size_in_bytes}")
	print(f"Number of vectors in the metadata: {len(vector_metadata)}")

	# NOTE:
	# 1,000 articles from 1 file generated over 8.65 KB of data. Each
	# file contains 150,000 artiles and file size averages around 750
	# MB:
	#
	# ~10 KB/article x 150,000 artices/file = 1,500,000 KB/file
	#                                       = 1,465 MB/file 
	#                  file vector metadata = 1.43 GB/file 
	# 
	# It is important to also consider how the metadata in each entry
	# of the vector metadata list is confined to a set size/limit.
	# Source file, article SHA1, embeddings, text chunk index, and text
	# chunk length are all confined to some defined upper limit
	# (depending on the attribute). For instance, the BERT embeddings 
	# will always be a 32-bit float of size (786), meaning there are
	# 786 32-bit float values in the embedding at all times. The only
	# thing that will affect the scaling of this data is the amount of
	# times the data is chunked (which usually corresponds to article
	# length but can also be influenced by unusually long words in the
	# text).
	# 
	# if the goal is to keep each index or table under a set size of 1
	# GB, I need to figure out how to chunk either the index itself or
	# the tables and iterate through those chunks to conduct a fast
	# search. 
	# Link to lancedb documentation: https://lancedb.github.io/lancedb/basic/

	###################################################################
	# VECTOR DB OPS
	###################################################################

	# Initialize vector DB.
	uri = "./data/lance_db_test"
	db = lancedb.connect(uri)

	# Get list of tables.
	table_names = db.table_names()
	print(F"Tables in {uri}: {', '.join(table_names)}")

	# Search for specified table if it is available.
	test_table_name = "test_table"
	table_in_db = test_table_name in table_names
	print(f"Searching for table '{test_table_name}' in database")
	print(f"Table found in database: {table_in_db}")

	# Create table if it does not exist in the database. Otherwise,
	# retrieve the table.
	if not table_in_db:
		table = db.create_table(test_table_name, data=vector_metadata)
	else:
		table = db.open_table(test_table_name)

	# Add the vectors to the table.
	table.add(data=vector_metadata)

	# Run search on the vector database. 
	# search_results = table.search(vector_metadata[0]["embedding"])\
	search_results = table.search(vector_metadata[0]["vector"])\
		.limit(2)
	print(json.dumps(search_results.to_list()))
	
	# Delete a value from the table.
	# table.delete(f'"SHA1" = {vector_metadata[-1]["SHA1"]}')

	# Delete the table.
	db.drop_table(test_table_name)

	# Exit the program.
	exit(0)


if __name__ == "__main__":
	main()