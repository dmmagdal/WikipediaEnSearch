# search.py
# A fast and lightweight program that allows for initializing and 
# reading an inverted index. Big Bucks Chungus Recommended this code
# as a way to keep resource usage low (for both file size and loading
# data into memory).
# Source: https://gist.github.com/sir-wabbit/d992fc1e3ecc9df29e114df1be797913
# Python 3.11
# Windows/MacOS/Linux

from typing import Tuple

import mmap
import struct
import os
from pathlib import Path
import re

from collections import Counter, defaultdict

class IndexFileReader:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        self.size = len(self.mmap)

        # Header:
        # int32 word_count; // total number of words
        # int32 word_offset[word_count + 1]; // offset of each word, last element is the end of the file

        self.word_count = struct.unpack('i', self.mmap[0:4])[0]
        self.header_end = 4 + 4 * (self.word_count + 1)

    def __del__(self):
        self.mmap.close()
        self.file.close()

    def get_word_pos(self, word_id) -> Tuple[int, int]:
        """
        Returns the offset and length of the word in the index file.
        """
        assert word_id < self.word_count, f"Word ID {word_id} is out of bounds (word count: {self.word_count})"
        assert word_id >= 0, f"Word ID {word_id} is out of bounds (word count: {self.word_count})"

        offset1 = struct.unpack('i', self.mmap[4 + 4 * word_id:4 + 4 * word_id + 4])[0]
        offset2 = struct.unpack('i', self.mmap[4 + 4 * (word_id + 1):4 + 4 * (word_id + 1) + 4])[0]
        return (offset1, offset2 - offset1)

    def documents(self, word_id):
        offset, length = self.get_word_pos(word_id)
        for _ in range(length):
            entry_offset = self.header_end + offset * 8
            doc_id = struct.unpack('i', self.mmap[entry_offset : entry_offset + 4])[0]
            doc_freq = struct.unpack('i', self.mmap[entry_offset + 4 : entry_offset + 8])[0]
            offset += 1
            yield (doc_id, doc_freq)

    def search(self, word_ids):
        word_ids = sorted(word_ids)
        T = len(word_ids)
        positions = [self.get_word_pos(word_id) for word_id in word_ids]
        offsets = [p[0] for p in positions]
        lengths = [p[1] for p in positions]
        indices = [0] * len(positions)

        # print(offsets, lengths, indices)
        while all(indices[i] < lengths[i] for i in range(T)):
            # print(offsets, lengths, indices)
            doc_ids = [self.get_doc_id(offsets[i], indices[i]) for i in range(T)]
            if all(doc_ids[i] == doc_ids[0] for i in range(len(doc_ids))):
                yield doc_ids[0]
                indices = [i + 1 for i in indices]
            else:
                min_index = min(range(len(doc_ids)), key=lambda i: doc_ids[i])
                indices[min_index] += 1

    def get_doc_id(self, offset, index):
        entry_offset = self.header_end + offset * 8
        entry_offset += index * 8
        return struct.unpack('i', self.mmap[entry_offset : entry_offset + 4])[0]


class IndexFileWriter:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w+b')
        self.word_documents = {} # word_id -> [(doc_id, doc_freq)]
        self.word_count = 0

    def __del__(self):
        self.file.close()

    def add_document(self, word_id, doc_id, doc_freq):
        if word_id not in self.word_documents:
            self.word_documents[word_id] = []
        self.word_count = max(self.word_count, word_id + 1)
        self.word_documents[word_id].append((doc_id, doc_freq))

    def write(self):
        word_count = len(self.word_documents)
        assert word_count == self.word_count, "Word count mismatch"
        key_list = sorted(self.word_documents.keys())
        assert key_list == list(range(word_count)), "Word IDs are not contiguous"

        assert word_count < 2**31, "Too many words"

        self.file.write(struct.pack('i', word_count))

        # Sort the documents by doc_id
        for word_id, documents in self.word_documents.items():
            self.word_documents[word_id] = sorted(documents, key=lambda x: x[0])

        offsets = [0]
        for word_id in key_list:
            offsets.append(offsets[-1] + len(self.word_documents[word_id]))
        for offset in offsets:
            self.file.write(struct.pack('i', offset))

        for word_id in key_list:
            for doc_id, doc_freq in self.word_documents[word_id]:
                self.file.write(struct.pack('i', doc_id))
                self.file.write(struct.pack('i', doc_freq))


SPLIT_REGEX = re.compile(r'\W+')
def tokens(text):
    text = text.strip()
    text = SPLIT_REGEX.split(text)
    text = [word.lower() for word in text if word]
    return text


def main():
    # Index each line in search.py as a separate document.

    # Step 1: Tokenize the documents
    with open(__file__) as f:
        lines = f.readlines()
        documents = [tokens(line) for line in lines]

    # Step 2: Count the frequency of each token
    token_freq = Counter()
    for doc_id, doc in enumerate(documents):
        for word in doc:
            token_freq[word] += 1

    # Step 3: Create a mapping from word to ID
    word2id = {word: i for i, (word, _) in enumerate(token_freq.most_common())}
    id2word = {i: word for word, i in word2id.items()}

    # Step 4: Create the inverted index
    index_file = 'index.bin'
    if os.path.exists(index_file):
        os.remove(index_file)
    writer = IndexFileWriter(index_file)
    for doc_id, doc in enumerate(documents):
        doc_freq = Counter(doc)
        for word, freq in doc_freq.items():
            writer.add_document(word2id[word], doc_id, freq)
    writer.write()
    del writer

    # Step 5: Read the inverted index
    reader = IndexFileReader(index_file)
    print(list(reader.search([word2id[word] for word in ['open', 'as']])))

if __name__ == '__main__':
    main()
