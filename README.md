# Wikipedia English Dataset Search

Description: Provides a text based search for Wikipedia (English only)


### Setup

 - Use either the dockerfile or a virtual environment to setup the necessary environment to run the files.
 - Python virtual environment (venv)
     - Initialize: `python -m venv wiki-search`
     - Activate: 
        - Windows: `/wiki-search/Scripts/activate`
        - Linux/MacOS: `source /wiki-search/bin/activate`
     - Install dependencies: `pip install -r requirements.txt`
 - Docker
     - Build: `docker build -t wiki-search -f Dockerfile .`
     - Run: `docker run -v {$pwd}:/wiki wiki-search`


### Usage

 - Preprocess Wikipedia data (`preprocess.py`)
     - `python preprocess.py`
 - Search Wikipedia (`searchwiki.py`)
     - `python searchwiki.py`


### Notes

 - Minimum system storage and memory requirements for each file: 
     - pages-articles-multistream.xml.bz2
         - 19+ GB disk space compressed
         - 86+ GB disk space decompressed
         - Expect to use at least 64+ GB of RAM for the decompression OR 8+ GB if using `--shard`
 - Preprocessing consists of the following:
     - Breaking each document (xml file/article) into a bag of words.
         - The bag of words is stored into a JSON dictionary.
         - For each word in the bag of words, extract the word count. This will make for faster compute by front loading that operation here. The word count is also stored in the bag of words JSON dictionary.
     - Chunking each document (xml file/article) into vector embeddings.
         - The vector embeddings are stored into a faiss vector index.
         - For each embedding, a mapping of the vectorDB, vectorDB index, file path, and slice indices are stored in a JSON dictionary.
 - Parsing documents
     - Read in the file and pass it to BeautifulSoup.
     - Extract only the `<title>` and `<text>` tags. Get the text for both tags and concatenate them together.
         - Ignore other tags (including `<links>`) as they don't have relevant information for the article.
     - For TF-IDF/BM25 search:
         - Split and/or replace all punctuation with "" (empty string) or " " (whitespace).
         - Tokenize all words (with nltk word_tokenize OR splitting on " " whitespace).
         - Lowercase all words.
         - Convert the list of all words into a set (bag of words).
         - Count the frequency of each word in the bag of words in the word list.
         - Store the word set (bag of words) and the count in a dictionary.
         - Store the dictionary in JSON.
     - For Vector search with language model:
         - Split the text on "\n" (newline) characters.
         - Embed the texts with the language model, maximizing the number of tokens that can fit within the model's context window. You will want to merge texts together (include the "\n" newline character you removed when you do).
         - Store the embedding to the vector database. Keep track of the index of that embedding as well as the text and article that it came from.
             - Store this metadata (database index, article file, text slice indices) in a JSON dictionary.
 - Parsing queries
     - Repeat the same steps from document parsing.
 - Search Methods
     - TF-IDF/BM25
         - Requires no additional modules but would benefit from having numpy installed for faster computation (minimal package bloat).
         - Is a more classical method for search.
         - Should at least give you the top article(s) to consider for each query/search.
     - Vector search with language model
         - Requires pytorch, transformers, and faiss libraries (meaning the package bloat will be heavier).
         - Is a more abstract and deep learning method of search.
         - Should give you the top passages of text to consider for each query/search.
         - Uses language models to turn articles and search queries into embeddings and performms k nearest neighbors (knn) to get the embeddings that are most semantically similar.
         - Models to be considered:
             - BERT
 - Nltk
     - Downloading all nltk packages (stored in `~/nltk_data`) results in 3.3 GB in storage.
 - Optimizations to consider for further development
     - In general
         - Parralellizing workflows (especially in preprocessing) to leverage multiprocessing/threading.
         - The embedding models are neural networks and thus would benefit from leveraging on device GPUs.
     - In preprocessing
         - Opting for utilizing a second machine or SQL database to load in article data/text from memory/RAM instead of using file IO to read xml files.
     - In search
         - Opting for utilizing a second machine to load in and query indices from memory/RAM instead of using file IO to read the index files.


### Useful Links for Preprocessing Concepts

 - Stop words
     - [pythonspot blog](https://pythonspot.com/nltk-stop-words/)
     - [geeksforgeeks](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)
 - Num2words
     - [pypi](https://pypi.org/project/num2words/)


### Python Module to JS Package

 - nltk
     - functions: stop words, lemmatization, stemming, and word tokenizing.
     - natural
     - winkjs
         - main [webpage](https://winkjs.org/packages.html)
 - num2word
     - functions: convert numbers from their numerical representation to a written form.
     - to-words
 - beautifulsoup
     - functions: parse xml data.


### References

 - BeautifulSoup [documentation](https://beautiful-soup-4.readthedocs.io/en/latest/)
 - NLTK [documentation](https://www.nltk.org/)
     - [corpus](https://www.nltk.org/api/nltk.corpus.html)
     - [download](https://www.nltk.org/api/nltk.downloader.html)
     - [stopwords](https://www.nltk.org/search.html?q=stopwords)
     - [tokenizer](https://www.nltk.org/api/nltk.tokenize.html)
     - [word_tokenize](https://www.nltk.org/api/nltk.tokenize.word_tokenize.html)
 - Documentation of native python module used:
     - [copy](https://docs.python.org/3.9/library/copy.html)
     - [json](https://docs.python.org/3.9/library/json.html)
     - [math](https://docs.python.org/3.9/library/math.html)
     - [os](https://docs.python.org/3.9/library/os.html)
     - [string](https://docs.python.org/3.9/library/string.html)