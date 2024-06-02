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

	# Load tokenizer and model.
	device = "mps"
	tokenizer, model = load_model(config, device)
	pipe = pipeline(
		"feature-extraction", model=model, tokenizer=tokenizer
	)

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
	print(f"unpadded len {len(unpadded_tokens)}")
	print(f"padded len {len(padded_tokens)}")
	print(padded_tokens)
	print(unpadded_tokens)
	assert len(unpadded_tokens) < len(padded_tokens), "Padded token sequence expected to be longer than unpadded"

	unpadded_text = tokenizer.decode(unpadded_tokens)
	padded_text = tokenizer.decode(
		padded_tokens, 
		skip_special_tokens=True
	) # See https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer.decode
	print(f"unpadded text ({len(unpadded_text)}): {unpadded_text}")
	print(f"padded text ({len(padded_text)}): {padded_text}")
	assert unpadded_text == padded_text, "Decoded texts were expected to be the same"

	# NOTE:
	# Current preprocessing (text chunking + feature extractioon/
	# embeddings) on M2 Macbook Pro is around 26 minutes for the slice
	# of pages that are currently in use. The script is using Pytorch's
	# MPS backend for Apple Silicon for the sake of faster embedding.

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

		# Embed the text chunks.
		for chunk in chunk_metadata:
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
				# print(embedding.shape)
				# print(torch.equal(embedding, output[1]))
				# exit()

	# Exit the program.
	exit(0)


if __name__ == "__main__":
	main()