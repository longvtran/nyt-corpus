#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 00:54:40 2018

@author: longtran
"""

import os
import hashlib
import struct
import collections
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

def chunk_file(set_name, finished_files_dir_path, chunks_dir_path):
    in_file = '%s.bin' % set_name
    reader = open(os.path.join(finished_files_dir_path, in_file), "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir_path, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all(finished_files_dir_path):
    chunks_dir_path = os.path.join(finished_files_dir_path, 'chunked')
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir_path):
        os.mkdir(chunks_dir_path)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name, finished_files_dir_path, chunks_dir_path)
    print("Saved chunked data in %s" % chunks_dir_path)


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def split_location(string, train_cutoff = 0.90, dev_cutoff = 0.05):
    """Using a heximal formated SHA1 hash of the input string, to decide
    whether the story goes to train, dev, or test set"""
    h = hashlib.sha1()
    h.update(string.encode())
    hash_val = h.hexdigest()
    
    # Divide the first 6 digits of the hash value to see where the story
    # should go
    split = int(hash_val[:6], 16) / 0xFFFFFF
    if split < train_cutoff:
        return "train"
    elif split < train_cutoff + dev_cutoff:
        return "val"
    else:
        return "test"


def get_art_abs(story_file, compress=False):
    lines = read_text_file(story_file)

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@abstract"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article, abstract


def write_to_bin(tokenize_dir_path, finished_files_dir_path, makevocab=True):
    """
    Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file.
    """
    
    # Get the total number of tokenized story files:
    num_tokenized_stories = sum([len(files) for r, d, files in os.walk(tokenize_dir_path)])
    
    years = os.listdir(tokenize_dir_path)
    if not os.path.exists(finished_files_dir_path): os.makedirs(finished_files_dir_path)
    
    if makevocab:
        vocab_counter = collections.Counter()
    
    story_index = 0
    for year in years:
        input_folder = os.path.join(tokenize_dir_path, year)
        stories = os.listdir(input_folder)
        
        for s in stories:
            split = split_location(s)
            out_file = ''.join([split, '.bin'])
            out_file_list = ''.join([split, '_list.txt'])

            with open(os.path.join(finished_files_dir_path, out_file), 'ab') as writer:
                if story_index % 1000 == 0:
                    print("Writing story %i of %i; %.2f percent done" % (story_index, num_tokenized_stories, float(story_index)*100.0/float(num_tokenized_stories)))
                
                story_file = os.path.join(input_folder, s)
                
                # Get the strings to write to .bin file
                article, abstract = get_art_abs(story_file)
                
                # Write to tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))
            
            with open(out_file_list, 'a') as list_writer:
                list_writer.write(s + "\n")
            
            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t!=""] # remove empty
                vocab_counter.update(tokens)
            
            story_index += 1
    
    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir_path, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':
    write_to_bin('/home/longtran/Stanford/2017-2018/CS224N/project/stories_tokenize_foo', '/home/longtran/Stanford/2017-2018/CS224N/project/finished_foo', True)
    chunk_all('/home/longtran/Stanford/2017-2018/CS224N/project/finished_foo')
    