# Notes

### Inverted Index

 - Inverted index takes a bag of words and returns the documents where those words were found in.
     - A blind inverted index returns around 10 million documents.
         - This is not optimal since words that are more common (but are not stop words) will pull in a lot of unrelated documents.
     - A few strategies have been devised to reduce the number of documents returned by the inverted index.
         - Weighted documents based on Inverse Document Frequency
             - For each document that is returned, the value associated with that document corresponds to the sum of all IDF values for the terms that returned the document. Documents with a weight above a certain threshold are included.
             - Pros
             - Cons
         - Ignore query words with low Inverse Document Frequency
             - Words with an IDF below a certain threshold are excluded from the blind inverted index.
             - Pros
             - Cons
         - Category traversal with weighted TF-IDF
             - Traverse the category graph based on the TF-IDF of the query and the words associated with the category.
                 - "A method for automated document classification using Wikipedia-derived weighted keywords" [Whitepaper](https://www.researchgate.net/publication/283227199_A_method_for_automated_document_classification_using_Wikipedia-derived_weighted_keywords) ([local copy](./Whitepapers/ICoDSE2014-AuthorCopy.pdf))
             - Pros
             - Cons
         - Category traversal with Word2Vec similarity
             - Pros
             - Cons


### Wikipedia Structure

 - [Help:Category](https://en.wikipedia.org/wiki/Help:Category)
 - [Wikipedia:Contents/Categories](https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories)
 - [Wikipedia:External search engines](https://en.wikipedia.org/wiki/Wikipedia:External_search_engines)
 - [Help:Searching](https://en.wikipedia.org/wiki/Wikipedia:External_search_engines)
 - [Wikicat repo](https://github.com/xhluca/wikicat)
     - Useful for managing and navigating graphs of Wikipedia categories


### Chunking

 - [Breaking up is hard to do: Chunking in RAG applications](https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications)