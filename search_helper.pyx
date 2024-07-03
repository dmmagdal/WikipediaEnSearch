# search_helper.pyx
# Cython file for defining certain helper functions primarily for 
# search.py aimed at improving speed on certain operations.
# Python 3.9
# Windows/MacOS/Linux


import cython
# from cython import dict, list, str, int
# from cython import list, str, int
# from cython import str, int
from cython import int
from libc.math import log
from typing import List, Dict


def compute_tf(doc_to_words: Dict, words: List[str], aggr: str = None):
    cdef int total_word_count, word_freq
    # cdef dict doc_tf = {}
    # cdef dict word_freq_map
    # cdef list word_vec = []
    # cdef str doc, word
    cdef char* doc, word
    doc_tf = dict()
    word_freq_map = dict()
    word_vec = []
    valid_aggr = ["sum", "mean"]

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

        # Set the text frequency word vector to the current document
        # (compute aggregation as necessary).
        if aggr is None or aggr not in valid_aggr:
            doc_tf[doc] = word_vec
        elif aggr == "add":
            doc_tf[doc] = sum(word_vec)
        elif aggr == "mean":
            doc_tf[doc] = sum(word_vec) / len(word_vec)
        
    # Return the dictionary of the document term frequency word
    # vectors.
    return doc_tf


def compute_idf():
    pass