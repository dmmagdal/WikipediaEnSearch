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
         - Expect to use at least 64+ GB of RAM for the decompression OR 8+ GB if using `--shard` when downloading the data.
 - Preprocessing consists of the following:
     - Breaking each document (xml file/article) into a bag of words.
         - The bag of words is stored into a JSON dictionary.
         - For each word in the bag of words, extract the word count. This will make for faster compute by front loading that operation here. The word count is also stored in the bag of words JSON dictionary.
     - Chunking each document (xml file/article) into vector embeddings.
         - The vector embeddings are stored into a faiss vector index.
         - For each embedding, a mapping of the vectorDB, vectorDB index, file path, and slice indices are stored in a JSON dictionary.
 - Nltk
     - Downloading all nltk packages (stored in `~/nltk_data`) results in 3.3 GB in storage.
     - NLTK uses WordNet Lemmatizer by default.
     - See the geeksforgeeks article on using NLTK's lemmatizer for the pros and cons of using this specific lemmatizer.
         - Advantages
             - Improves text analysis accuracy: Lemmatization helps in improving the accuracy of text analysis by reducing words to their base or dictionary form. This makes it easier to identify and analyze words that have similar meanings.
             - Reduces data size: Since lemmatization reduces words to their base form, it helps in reducing the data size of the text, which makes it easier to handle large datasets.
             - Better search results: Lemmatization helps in retrieving better search results since it reduces different forms of a word to a common base form, making it easier to match different forms of a word in the text.
         - Disadvantages
             - Time-consuming: Lemmatization can be time-consuming since it involves parsing the text and performing a lookup in a dictionary or a database of word forms.
             - Not suitable for real-time applications: Since lemmatization is time-consuming, it may not be suitable for real-time applications that require quick response times.
             - May lead to ambiguity: Lemmatization may lead to ambiguity, as a single word may have multiple meanings depending on the context in which it is used. In such cases, the lemmatizer may not be able to determine the correct meaning of the word.
 - Optimizations to consider for further development
     - In general
         - Parralellizing workflows (especially in preprocessing) to leverage multiprocessing/threading.
         - The embedding models are neural networks and thus would benefit from leveraging on device GPUs.
     - In preprocessing
         - Opting for utilizing a second machine or SQL database to load in article data/text from memory/RAM instead of using file IO to read xml files.
     - In search
         - Opting for utilizing a second machine to load in and query indices from memory/RAM instead of using file IO to read the index files.
 - Multiprocessing
     - Getting transformer models running with multiprocessing
         - Cannot re-initialize CUDA in forked subprocess [github issue](https://github.com/pytorch/pytorch/issues/40403#issuecomment-648378957)
     - Videos going using python's multiprocessing
         - Unlocking your CPU cores in Python (multiprocessing) [video](https://www.youtube.com/watch?v=X7vBbelRXn0)
         - Multiprocessing in Python: Pool [video](https://www.youtube.com/watch?v=u2jTn-Gj2Xw)
         - Python starmap (itertools): A short and easy intro [video](https://www.youtube.com/watch?v=wiwb5WAByFE)
 - Got ascii art from [here](https://patorjk.com/software/taag/#p=display&f=Sub-Zero&t=Wikipedia%20%0ASearch)


### Parsing Documents & Queries

 - Read in the file and pass it to BeautifulSoup.
 - Extract only the `<title>` and `<text>` tags. Get the text for both tags and concatenate them together.
     - Ignore other tags (including `<links>`) as they don't have relevant information for the article.
 - For TF-IDF/BM25 search:
     - Preprocessing:
         - Lowercase all words.
         - Split and/or replace all punctuation with "" (empty string) or " " (whitespace).
         - Tokenize all words (with nltk word_tokenize OR splitting on " " whitespace).
         - Convert all numbers from digits to their written expanded form (1 -> 'one').
         - Lemmatize and stem all remainging words.
     - Convert the list of all words into a set (bag of words).
     - Document to words mapping:
         - Each document is going to contain its own list of the set of unique words (bag of words).
         - For each word in the document's bag of words, the word frequency will be kept track of.
         - 
```
document_path: {
    word_1: word_1_freq,
    word_2: word_2_freq,
    . . .
    word_n: word_n_freq,
}
```
     - Word to documents mapping:
         - Each word is going to maintain a list of the documents that it has occured in.
         - 
```
word: [document_1_path, document_2_path, ... , document_n_path]
```
     - Store the above metadata into dictionaries and save them to JSON files.
 - For Vector search with language model:
     - Preprocessing
         - Created a preprocessor the recursively splits the text into paragraphs, words, and characters.
             - Paragraph splitter splits on newline characters ("\n\n", "\n").
             - Word splitter splits on white space characters (" ").
             - Character splitter splits on characters ("" or empty strin).
         - When a text segment is too large for the embedding model, it is either passed to the next level splitter or further broken down on the current splitter.
         - Inspiration for the design of the recursive splitter is the Langchain `RecursiveTextCharacterSplitter`.
             - All links below are for v0.1 of Langchain.
             - Langchain RecursiveCharacterTextSplitter in the [documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)
             - Langchain RecursiveCharacterTextSplitter implementation on [github](https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py)
         - Preprocessor captures subtext token sequence, index with respect to original text, and subtext length along with the article's SHA1 hash and the xml file it came from.
     - Embed the texts with the language model, maximizing the number of tokens that can fit within the model's context window. The subtext metadata for each embedded chunk will come in handy for retrieval.
     - Store the embedding to the vector database. Keep track of the index of that embedding as well as the text and article that it came from.
         - Store this metadata (database index, article file, text slice indices) in a JSON dictionary.
         - 
```
[database_path, database_index, article_path, [text_slice_start_index, text_slice_end_index]]
```
 - Repeat the same steps from document parsing to parse search queries.


### Search Methods

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
             - bert-base-cased
             - mobilebert
         - All MiniLM ([sentence-transformers](https://huggingface.co/sentence-transformers))
             - sentence-transformers/all-MiniLM-L6-v2
             - sentence-transformers/all-MiniLM-L6-v1
             - sentence-transformers/all-MiniLM-L12-v2
             - sentence-transformers/all-MiniLM-L12-v1
 - Rerank search with TF-IDF/BM25 and vector search
     - Combination of methods above.


### Useful Links for Preprocessing Concepts

 - Stop words
     - [pythonspot blog](https://pythonspot.com/nltk-stop-words/)
     - [geeksforgeeks](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)
 - Num2words
     - [pypi](https://pypi.org/project/num2words/)
 - Lemmatization
     - [geeksforgeeks](https://www.geeksforgeeks.org/python-lemmatization-with-nltk/)
 - Stemming
     - [geeksforgeeks](https://www.geeksforgeeks.org/python-stemming-words-with-nltk/)
     - [guru99 post](https://www.guru99.com/stemming-lemmatization-python-nltk.html) (applies also to lemmatization)
 - Previous search examples
     - neuralsearch [repo](https://github.com/dmmagdal/NeuralSearch) (mine)


### Python Module to JS Package

 - nltk
     - functions: stop words, lemmatization, stemming, and word tokenizing.
     - natural
         - [npm](https://www.npmjs.com/package/natural/v/1.0.1)
         - main [webpage](https://naturalnode.github.io/natural/)
     - winkjs
         - wink-nlp [npm](https://www.npmjs.com/package/wink-nlp)
         - wink-porter2-stemmer [npm](https://www.npmjs.com/package/wink-porter2-stemmer)
         - wink-nlp-utils [npm](https://www.npmjs.com/package/wink-nlp-utils)
         - wink-nlp-lemmatizer [npm](https://www.npmjs.com/package/wink-lemmatizer)
         - main [webpage](https://winkjs.org/packages.html)
         - examples [page](https://winkjs.org/examples.html)
         - stemming and lemmatization [tutorial](https://observablehq.com/@winkjs/how-to-do-stemming-and-lemmatization)
         - wink-nlp main [webpage](https://winkjs.org/wink-nlp/)
         - wink-nlp-utils main [webpage](https://winkjs.org/wink-nlp-utils/)
 - num2word
     - functions: convert numbers from their numerical representation to a written form.
     - to-words
         - [npm](https://www.npmjs.com/package/to-words)
 - beautifulsoup
     - functions: parse xml data.
     - some alternatives (curtesy of [stack overflow](https://stackoverflow.com/questions/14890655/the-best-node-module-for-xml-parsing))
     - libxmljs
         - [npm](https://www.npmjs.com/package/libxmljs)
         - [github](https://github.com/libxmljs/libxmljs)
     - xml-stream
         - [npm](https://www.npmjs.com/package/xml-stream)
         - [github](https://github.com/assistunion/xml-stream)
     - xmldoc
         - [npm](https://www.npmjs.com/package/xmldoc)
         - [github](https://github.com/nfarina/xmldoc)
 - faiss 
     - functions: store/retrieve and save/load vector embeddings.
     - faiss-node
         - [npm](https://www.npmjs.com/package/faiss-node)
 - chromadb
     - functions: store/retrieve and save/load vector embeddings
     - chromadb
         - [npm](https://www.npmjs.com/package/chromadb)
 - transformers
     - functions: use pretrained language models to create vector embeddings from texts.
     - transformers.js
         - [npm](https://www.npmjs.com/package/@xenova/transformers)


### References

 - BeautifulSoup [documentation](https://beautiful-soup-4.readthedocs.io/en/latest/)
 - Faiss [documentation](https://faiss.ai/index.html)
     - [wiki](https://github.com/facebookresearch/faiss/wiki)
 - NLTK [documentation](https://www.nltk.org/)
     - [corpus](https://www.nltk.org/api/nltk.corpus.html)
     - [download](https://www.nltk.org/api/nltk.downloader.html)
     - [stopwords](https://www.nltk.org/search.html?q=stopwords)
     - [tokenizer](https://www.nltk.org/api/nltk.tokenize.html)
     - [word_tokenize](https://www.nltk.org/api/nltk.tokenize.word_tokenize.html)
     - [wordnet lemmatizer](https://www.nltk.org/api/nltk.stem.wordnet.html)
     - [porter stemmer](https://www.nltk.org/howto/stem.html)
 - Transformers
     - getting started with embeddings [blog post](https://huggingface.co/blog/getting-started-with-embeddings)
     - using sentencetransformers at huggingface [documentation](https://huggingface.co/docs/hub/en/sentence-transformers)
     - loading models direcdtly to gpu [stackoverflow](https://stackoverflow.com/questions/77237818/how-to-load-a-huggingface-pretrained-transformer-model-directly-to-gpu)
     - estimate model memory usage [documentation](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator)
 - Sentence Transformers
     - sentence transformers in the huggingface hub [blog post](https://huggingface.co/blog/sentence-transformers-in-the-hub)
     - [documentation](https://www.sbert.net/)
     - [github](https://github.com/UKPLab/sentence-transformers)
 - Documentation of native python module used:
     - [argparse](https://docs.python.org/3.9/library/argparse.html)
     - [copy](https://docs.python.org/3.9/library/copy.html)
     - [gc](https://docs.python.org/3.9/library/gc.html)
     - [json](https://docs.python.org/3.9/library/json.html)
     - [math](https://docs.python.org/3.9/library/math.html)
     - [multiprocessing](https://docs.python.org/3.9/library/multiprocessing.html)
     - [os](https://docs.python.org/3.9/library/os.html)
     - [string](https://docs.python.org/3.9/library/string.html)
     - [shutil](https://docs.python.org/3.9/library/shutil.html)
     - [typing](https://docs.python.org/3.9/library/typing.html)