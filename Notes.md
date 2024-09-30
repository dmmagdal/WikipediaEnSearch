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


### Wikipedia Structure

 - [Help:Category](https://en.wikipedia.org/wiki/Help:Category)
 - [Wikipedia:Contents/Categories](https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories)
 - [Wikipedia:External search engines](https://en.wikipedia.org/wiki/Wikipedia:External_search_engines)
 - [Help:Searching](https://en.wikipedia.org/wiki/Wikipedia:External_search_engines)
 - [Wikicat repo](https://github.com/xhluca/wikicat)
     - Useful for managing and navigating graphs of Wikipedia categories


### Chunking

 - [Breaking up is hard to do: Chunking in RAG applications](https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications)


### Inverted Index

 - [Inverted Index: a simple yet powerful data structure](https://evanxg852000.github.io/tutorial/rust/data/structure/2020/04/09/inverted-index-simple-yet-powerful-ds.html)
     - Inverted index implemented in rust


### Doc2Vec

 - [Doc2Vec: Understanding Document Embeddings for Natural Language Processing](https://medium.com/@evertongomede/doc2vec-understanding-document-embeddings-for-natural-language-processing-ba244e55eba3)
     - Use freedium since this is a premium medium article