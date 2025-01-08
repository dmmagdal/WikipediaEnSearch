# Notes

### Inverted Index

 - Inverted index takes a bag of words and returns the documents where those words were found in.
     - A blind inverted index returns around 10 million documents.
         - Pros
             - Completely exhaustive of the dataset.
             - Simple to implement with good compression (data storage) via the trie structure.
         - Cons
             - Returns too many documents. Use of set union results in most of the original corpus being returned for words that are common but not stop words.
     - A few strategies have been devised to reduce the number of documents returned by the inverted index.
         - Weighted documents based on Inverse Document Frequency
             - For each document that is returned, the value associated with that document corresponds to the sum of all IDF values for the terms that returned the document. Documents with a weight above a certain threshold are included.
             - Pros
                 - Simple to implement with good compression (data storage) via the trie structure.
                 - Eliminates a lot of documents (depending on the threshold).
             - Cons
                 - Not as exhaustive as the blind inverted index. May skip some relevant documents.
                 - Threshold requires finetuning via trial & error.
         - Ignore query words with low Inverse Document Frequency
             - Words with an IDF below a certain threshold are excluded from the blind inverted index.
             - Pros
                 - Simple to implement with good compression (data storage) via the trie structure.
                 - Eliminates a lot of common words which in turn eliminates a lot of documents (depending on the threshold).
             - Cons
                 - Not as exhaustive as the blind inverted index. May skip some relevant documents.
                 - Threshold requires finetuning via trial & error.
         - Category traversal with weighted TF-IDF
             - Traverse the category graph based on the TF-IDF of the query and the words associated with the category.
                 - "A method for automated document classification using Wikipedia-derived weighted keywords" [Whitepaper](https://www.researchgate.net/publication/283227199_A_method_for_automated_document_classification_using_Wikipedia-derived_weighted_keywords) ([local copy](./Whitepapers/ICoDSE2014-AuthorCopy.pdf))
             - Pros
                 - More abstract but also more direct/faster than plain inverted index.
             - Cons
                 - More storage usage than blind inverted index. Lots of redundant/overlapping mappings between each node in the category tree.
                 - Compute heavy. Requires computing a TF-IDF at runtime for each level in the tree structure.
         - Category traversal with Word2Vec similarity
             - Pros
                 - More abstract but also more direct/faster than plain inverted index.
             - Cons
                 - Heavy reliance on third party structures/packages.
                     - Requires word2vec model
                         - May or may not be supported for other languages (thus locking this part of the code to python).
                         - Model is pretrained on some other dataset. May be a good idea to look at a model trained on wikipedia or train it myself.
                     - Wikicat repo below for simplified creation/structuring of the categories.
                         - Is a python-only repo/package.
                     - Doesn't handle multi word labels (can only convert one word at a time to vector).
         - Category traversal with BERT similarity
             - Pros
                 - More abstract but also more direct/faster than plain inverted index.
             - Cons
                 - Compute intensive
                     - Every query will have to be embedded (cost is a few seconds but that is per query).
                     - Will have to precompute vectors for categories and store that information.
                         - Takes time depending on the hardware where this is being deployed.
                         - Incurs storage cost too.
                     - Have to recompute every time the embedding model is switched.
         - Document vectors with Doc2Vec
             - Pros
                 - Compresses documents to small vectors.
             - Cons
                 - Compute intensive
                     - Will take a while to build and train model.
                     - Will have to precompute vectors for categories and store that information.
                         - Takes time depending on the hardware where this is being deployed.
                         - Incurs storage cost too.
                 - Scaling
                     - Will have to scale to all documents (roughly 26 to 40 million). That will incur a lot of time when computing similarity and ranking. No better than the results that were being returned from the blind inverted index (possibly worse).
         - Directy category mapping with BERT similarity
             - Similar to "Category traversal with BERT similarity" except no category tree is involved.
             - Pros:
                 - No category tree means more direct access to end categories.
                 - Let's the vector DB do all the work in one shot.
             - Cons:
                 - Does not work with raw text queries like "What color are sapphires?".
 - [Inverted Index: a simple yet powerful data structure](https://evanxg852000.github.io/tutorial/rust/data/structure/2020/04/09/inverted-index-simple-yet-powerful-ds.html)
     - Inverted index implemented in rust


### Wikipedia Structure

 - [Help:Category](https://en.wikipedia.org/wiki/Help:Category)
 - [Wikipedia:Contents/Categories](https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories)
 - [Wikipedia:External search engines](https://en.wikipedia.org/wiki/Wikipedia:External_search_engines)
 - [Help:Searching](https://en.wikipedia.org/wiki/Wikipedia:External_search_engines)
 - [Wikipedia:Categories](https://simple.wikipedia.org/wiki/Wikipedia:Categories)
     - Contains information on Wikipedia's category tree
 - [Category:Commons category link on Wikipedia](https://en.wikipedia.org/wiki/Category:Commons_category_link_is_on_Wikidata)
     - Contains information on commons category links in Wikipedia
 - [Extension:Category Tree](https://www.mediawiki.org/wiki/Extension:CategoryTree)
     - Contains information on Wikipedia's category tree
 - [Wikipedia:Vital Articles](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles)
     - Could also be used for creating a category tree
 - [Category:Main topic classifications](https://en.wikipedia.org/wiki/Category:Main_topic_classifications)
     - Used as the root node for creating the category tree
 - [Manual:categorylinks table](https://www.mediawiki.org/wiki/Manual:Categorylinks_table)
 - [API:Main page](https://www.mediawiki.org/wiki/API:Main_page)
 - Extract Wikipedia categories from dump - [stackoverflow](https://stackoverflow.com/questions/17432254/wikipedia-category-hierarchy-from-dumps)
 - Build Wikipedia category tree from dump - [stackoverflow](https://stackoverflow.com/questions/27279649/how-to-build-wikipedia-category-hierarchy)
 - Parsing Wikipedia Page Hierarchy - [Koding Notes](https://kodingnotes.wordpress.com/2014/12/03/parsing-wikipedia-page-hierarchy/)
     - Deals directly with category links stored in mySQL (`.sql`) file.
 - What Wikipedia’s Network Structure Can Tell Us About Culture - [medium article](https://docmarionum1.medium.com/what-wikipedias-network-structure-can-tell-us-about-culture-38f8caabf69d)
     - Not directly applicable to the research but was still interesting to read.
 - [Wikicat repo](https://github.com/xhluca/wikicat)
     - Useful for managing and navigating graphs of Wikipedia categories


### Graph Theory

 - MAT202: Introduction to Discrete Mathematics - [isomorphisms and Subgraphs](https://tjyusun.com/mat202/sec-graph-isomorphisms.html)
     - Useful for graph theory and understanding isomorphism vs graph equality.


### Chunking

 - [Breaking up is hard to do: Chunking in RAG applications](https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications)


### Embedding

 - [Binary and scalar embedding quantization for significantly faster & cheaper retrieval](https://huggingface.co/blog/embedding-quantization)


### Doc2Vec

 - [Gensim Doc2Vec Documentation](https://radimrehurek.com/gensim/models/doc2vec.html)
 - [Doc2Vec: Understanding Document Embeddings for Natural Language Processing](https://medium.com/@evertongomede/doc2vec-understanding-document-embeddings-for-natural-language-processing-ba244e55eba3)
     - Use freedium since this is a premium medium article


### Keyword Extraction

 - Keyphrase Extraction in NLP ([geeksforgeeks](https://www.geeksforgeeks.org/keyphrase-extraction-in-nlp/))
 - Keyword Extraction Methods in NLP ([geeksforgeeks](https://www.geeksforgeeks.org/keyword-extraction-methods-in-nlp/))
 - Four of the easiest and most effective methods to Extract Keywords from a Single Text using Python ([Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/01/four-of-the-easiest-and-most-effective-methods-of-keyword-extraction-from-a-single-text-using-python/))
 - Fast and Effective ways to Extract Keyphrases using TFIDF with Python ([Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/12/how-to-extract-key-phrases-using-tfidf-with-python/))
 - Keyword Extraction Methods from Documents in NLP ([Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/03/keyword-extraction-methods-from-documents-in-nlp/))
 - Keyword Extraction process in Python with Natural Language Processing(NLP) ([Medium](https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c))
 - [Top 5 Keyword Extraction Algorithms in NLP](https://www.analyticssteps.com/blogs/top-5-keyword-extraction-algorithms-nlp)
 - Rust
     - Keyword extraction in rust ([GitHub](https://github.com/tugascript/keyword-extraction-rs))
 - How To Implement Keyword Extraction [3 Ways In Python With NLTK, SpaCy & BERT] ([blog site](https://spotintelligence.com/2022/12/13/keyword-extraction/))
 - Algorithms
     - Rake
         - [Understanding the RAKE Algorithm in 2024: A Simple Guide](https://www.markovml.com/blog/rake-algorithm)
         - RAKE Explained with python Implementation ([Medium](https://medium.com/@bharathwajan/rake-explained-with-python-implementation-95ecaeb6c580))
         - Rapid Keyword Extraction (RAKE) Algorithm in Natural Language Processing ([Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/10/rapid-keyword-extraction-rake-algorithm-in-natural-language-processing/))
         - Extracting Keyphrases from Text: RAKE and Gensim in Python ([Medium](https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f))
         - [Unsupervised Key Phrases (Topic) Extraction with RAKE | Applied NLP Tutorial in Python](https://www.youtube.com/watch?v=ZOgrhn2Uq0U&ab_channel=1littlecoder)
         - Harnessing the Power of LLMs for Keyword Extraction with RAKE: A Python Sample Guide ([Medium](https://medium.com/@amb39305/harnessing-the-power-of-llms-for-keyword-extraction-with-rake-a-python-sample-guide-a305608bc332))
     - Yake
         - [YAKE Keyword Analysis: A Simplified Guide for NLP Enthusiasts](https://www.markovml.com/blog/yake-keyword-extraction)
         - [Rust Keyword Extraction: Creating the YAKE! algorithm from scratch](https://dev.to/tugascript/rust-keyword-extraction-creating-the-yake-algorithm-from-scratch-4n2l)
     - Textrank
         - Two minutes NLP — Keyword and keyphrase extraction with PKE ([Medium](https://medium.com/nlplanet/two-minutes-nlp-keyword-and-keyphrase-extraction-with-pke-5a0260e75f3e))
         - Implementation of TextRank Algorithm Methods for Keyword Extraction ([Medium](https://medium.com/@theofany007/implementation-of-textrank-and-methods-for-keyword-extraction-b84f8f145b2e))
         - Decomposing the TextRank algorithm, to grasp the main idea behind it + code implementation ([Medium](https://medium.com/@fabiosalern/decomposing-the-textrank-algorithm-to-grasp-the-main-idea-behind-it-code-implementation-b29414eba821))
         - Math behind TextRank Algorithm ([Medium](https://ankitnitjsr13.medium.com/text-rank-algorithm-a8c2cc58ea9c))
         - An Introduction to Text Summarization using the TextRank Algorithm (with Python implementation) ([Analytics Vidhya](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/))
         - [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
     - KeyBERT
         - Keyphrase Extraction with BERT Transformers and Noun Phrases ([Medium](https://towardsdatascience.com/enhancing-keybert-keyword-extraction-results-with-keyphrasevectorizers-3796fa93f4db))


### Similar Works

 - [Developing a Search Engine Framework for Wikipedia Articles](https://medium.com/@akleber/developing-a-search-engine-framework-for-wikipedia-articles-81fcbd95a928)
 - [Improving Vector Search to Find the Most Relevant Documents](https://medium.com/@PascalBiese/improving-vector-search-to-find-the-most-relevant-papers-ce6b6d4222f1)
 - [Multilingual GPU-Powered Topic Modeling at Scale](https://medium.com/bumble-tech/multilingual-gpu-powered-topic-modelling-at-scale-dc8bd08609ef)
     - RAPIDs package [homepage](https://rapids.ai/)
 - [Hybrid Search: Combining BM25 and Semantic Search for Better Results with Langchain](https://medium.com/etoai/hybrid-search-combining-bm25-and-semantic-search-for-better-results-with-lan-1358038fe7e6)
 - [Building a better RAG: A practical guide to two-step retrieval with langchain](https://medium.com/@alihaydargulec/building-a-better-rag-a-practical-guide-to-two-step-retrieval-with-langchain-e9ffe6e8aa8b)
 - [Rerankers and two-stage retrieval](https://www.pinecone.io/learn/series/rag/rerankers/)
 - [What is RAG? Enhancing LLMs with dynamic information access](https://sendbird.com/developer/tutorials/rag)


### Python Acceleration with Rust

 - [How to use Rust with Python, and Python with Rust](https://www.infoworld.com/article/2335770/how-to-use-rust-with-python-and-python-with-rust.html)
 - [How to use PyO3 to write Python extensions in Rust](https://www.youtube.com/watch?v=fgC8YxNwBfQ)
 - [How To Make Your Python Packages Really Fast With RUST](https://www.youtube.com/watch?v=jlWhnrk8go0)
 - [Which programming language is faster at reading?](https://dev.to/fredysandoval/which-programming-language-is-faster-at-reading-10gn)
 - RusTy - Rust bindings for SpaCy ([GitHub](https://github.com/dluman/rusTy))
 - [Rust Keyword Extraction: Creating the YAKE! algorithm from scratch](https://dev.to/tugascript/rust-keyword-extraction-creating-the-yake-algorithm-from-scratch-4n2l)


### B-Trees

 - Introduction of B-Tree ([geeksforgeeks](https://www.geeksforgeeks.org/introduction-of-b-tree-2/))
 - B-Tree ([wikipedia](https://en.wikipedia.org/wiki/B-tree))


### NLP Frameworks/Libraries in Other Languages

 - C++
     - [8 Excellent C++ Natural Language Processing Tools](https://www.linuxlinks.com/excellent-c-plus-plus-natural-language-processing-tools/)
         - [MITIE: MIT Information Extraction](https://www.linuxlinks.com/mitie-mit-information-extraction/)
         - [MeTA – modern C++ data sciences toolkit](https://www.linuxlinks.com/meta-modern-c-plus-plus-data-sciences-toolkit/)
     - Facebook AI Research [fastText repo](https://github.com/facebookresearch/fastText)