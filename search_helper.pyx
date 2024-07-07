# search_helper.pyx
# Cython file for defining certain helper functions primarily for 
# search.py aimed at improving speed on certain operations.
# Python 3.9
# Windows/MacOS/Linux


import json
from typing import List, Dict

import cython
# from cython cimport array
# from cython import dict, list, str, int
# from cython import list, str, int
# from cython import str, int
from cython import int
from libc.math import log
# from libc.stdlib import malloc, free
import msgpack


# NOTE: 
# For functions associated with computing TF-IDF, it is assume that
# the words list argument is sorted/fixed so as to maintain consistent
# ordering for the output vector. For instance, if the words list is
# ['happy', 'go', 'lucky'], then the resulting vectors map to the same
# index of the respective words (ie 'happy' always maps to the 0 index
# in the vectors, 'go' always maps to the 1 index in the vectors, and
# so on).


def compute_tf(doc_to_words: Dict, words: List[str]):
    cdef int total_word_count, word_freq
    # cdef dict doc_tf = {}
    # cdef dict word_freq_map
    # cdef list word_vec = []
    # cdef str doc, word
    cdef char* doc, word
    doc_tf = dict()
    word_freq_map = dict()
    word_vec = []

    # Iterate through each document.
    for doc in doc_to_words:
        # Initialize the document's word vector.
        word_vec = []

        # Extract the document owrd frequencies.
        word_freq_map = doc_to_words[doc]

        # Compute total word count.
        total_word_count = sum(
            [value for value in word_freq_map.values()]
        )

        # Compute the term frequency accordingly and add it to the 
        # document's word vector
        for word in words:
            if word in word_freq_map:
                word_freq = word_freq_map[word]
                word_vec.append(word_freq / total_word_count)
            else:
                word_vec.append(0)
        
    # Return the dictionary of the document term frequency word
    # vectors.
    return doc_tf


def compute_idf():
    pass


def compute_tfidf(doc_to_words: Dict, words: List[str], srt: float = -1.0):
    # cdef int size = len(words)
    # cdef cython.float *array = <float *>malloc(size * sizeof(cython.float))

    # Compute IDF.
    idf = compute_idf()

    # Iterate through each document.
    for doc in doc_to_words.keys():
        word_freq = doc_to_words[doc]

        # Compute total document length.
        doc_len = sum([value for value in word_freq.values()])

        word_vec = [0] * len(words)

        for word_idx in range(len(words)):
            word = words[word_idx]

            # Compute term frequency.
            word_tf = word_freq[word] / doc_len

            # Compute TF-IDF for word.
            word_tfidf = word_tf * idf
        
        # Compute document cosine similarity given current TF-IDF
        # vector.

    pass