# summarize_articles.py
# Leverage a small LLM like Flan-T5/T5 or Llama 3.2 1B to summarize 
# wikipedia articles. These summaries can be used further down the line
# as abstracts.
# Source: https://github.com/dmmagdal/HuggingfaceOfflineDownloader/
#   blob/main/huggingface_models/chat_inference.py
# Python 3.9
# Windows/MacOS/Linux


import os

from bs4 import BeautifulSoup
import torch
from tqdm import tqdm
from transformers import pipeline


def main():

	file_path = "./WikipediaEnDownload/WikipediaData"
	doc_files = [
		os.path.join(file_path, file)
		for file in sorted(os.listdir(file_path))
		if file.endswith(".xml")
	]
	doc_files = [doc_files[0]] # NOTE: Only chosing to do 1 file for now.

	# Determine device (cpu, mps, or cuda).
	device = 'cpu'
	if torch.backends.mps.is_available():
		# device = 'mps' # mps device causes errors and more memory usage for tinyllama while cpu doesnt
		device = 'cpu'
	elif torch.cuda.is_available():
		device = 'cuda'

	# Model ID (variant of model to load).
	model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	# model_id = "./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0"
	model_id = "meta-llama/Llama-3.2-1B-Instruct"
	# model_id = "./models/meta-llama_Llama-3.2-1B-Instruct"
	if not os.path.exists("../.env"):
		print("Path to .env file with huggingface token was not found")
		exit(1)
	with open("../.env", "r") as f:
		token = f.read().strip("\n")
	
	# Initialize tokenizer & model.
	# tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)#, device_map="auto")
	#model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
	#model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16)
	#model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
	# model = AutoModelForCausalLM.from_pretrained(model_id)
	#model = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype=torch.float16)
	pipe = pipeline("text-generation", model_id, token=token, device=device)

	# Pass model to device.
	# model = model.to(device)

	# Initial chat template to pass into the model
	messages = [
		{
			"role": "system",
			"content": "You are a friendly chatbot who always responds to any query or question asked",
		},
		{
			"role": "user",
			"content": "",
		}
	]

	# Iterate through each document in the dataset.
	for file in doc_files:
		print(f"Summarizing articles from {os.path.basename(file)}...")

		# Load in file.
		with open(file, "r") as f:
			raw_data = f.read()

		# Parse file with beautifulsoup. Isolate the articles.
		soup = BeautifulSoup(raw_data, "lxml")
		pages = soup.find_all("page")[:10] # NOTE: Only chosing to do 10 articles for now.

		# Iterate through each article.
		for page in tqdm(pages):
			###########################################################
			# EXTRACT TEXT
			###########################################################

			# Isolate the article/page's SHA1.
			sha1_tag = page.find("sha1")

			# Skip articles that don't have a SHA1 (should not be 
			# possible but you never know).
			if sha1_tag is None:
				continue

			# Clean article SHA1 text.
			article_sha1 = sha1_tag.get_text()
			article_sha1 = article_sha1.replace(" ", "").replace("\n", "")

			# Isolate the article/page's redirect tag.
			redirect_tag = page.find("redirect")

			# Skip articles that have a redirect tag (they have no 
			# useful information in them).
			if redirect_tag is not None:
				continue

			# Compute the file hash.
			# file_hash = file + article_sha1
			file_hash = os.path.basename(file) + article_sha1

			# Extract the article text.
			title_tag = page.find("title")
			text_tag = page.find("text")

			# Verify that the title and text data is in the article.
			assert None not in [title_tag, text_tag]

			# Extract the text from each tag.
			title = title_tag.get_text()
			title = title.replace("\n", "").strip()
			text = text_tag.get_text().strip()
			full_text = title + "\n\n" + text

			###########################################################
			# PROMPT MODEL
			###########################################################

			# Insert the wikipedia text into the prompt.
			messages[-1]["content"] = f"Write a short summary of the wikipedia article provided below: \n\n {full_text}"

			# Apply the prompt as a chat template. Pass the template prompt
			# to the model and print the output.
			prompt = pipe.tokenizer.apply_chat_template(
				messages, tokenize=False, add_generation_prompt=True
			)
			outputs = pipe(
				prompt, 
				max_new_tokens=2048,
				do_sample=True,
				temperature=0.7,
				# top_k=50, # from tinyllama chat v1.0 model card. Gave "RuntimeError: Currently topk on mps works only for k<=16"
				top_k=1,
				top_p=0.95
			)
			# print(outputs[0]["generated_text"])
			with open(f"{file_hash}_summary.txt", "w+") as f:
				f.write(outputs[0]["generated_text"])

			# Another reference: https://rumn.medium.com/setting-top-k-top-p-and-temperature-in-llms-3da3a8f74832

			# Tokenize and process the text in the model. Print the output.
			# input_ids = tokenizer(
			# 	text_input, 
			# 	return_tensors='pt'
			# ).input_ids.to(device)
			# # output = model.generate(input_ids, max_length=512)
			# output = model.generate(
			# 	input_ids, 
			# 	do_sample=True,
			# 	min_length=64,				# default 0, the min length of the sequence to be generated (corresponds to input_prompt + max_new_tokens)
			# 	max_length=512,				# default 20, the max langth of the sequence to be generated (corresponds to input_prompt + max_new_tokens)
			# 	length_penalty=2,			# default 1.0, exponential penalty to the length that is used with beam based generation
			# 	temperature=0.7,			# default 1.0, the value used to modulate the next token probabilities
			# 	num_beams=16, 				# default 4, number of beams for beam search
			# 	no_repeat_ngram_size=3,		# default 3
			# 	early_stopping=True,		# default False, controls the stopping condition for beam-based methods
			# )	# more detailed configuration for the model generation parameters. Depending on parameters, may cause OOM. Should play around with them to get desired output.
			# print(tokenizer.decode(output[0], skip_special_tokens=True))


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()