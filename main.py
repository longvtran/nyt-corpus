#! /usr/bin/env python3
# Author: Long Tran (longtran@stanford.edu)

import argparse
from make_stories import tokenize_all_docs
from make_bins import write_to_bin, chunk_all


def add_args(parser):
    """Command-line arguments to extract a summarization dataset from the
    NYT Annotated Corpus.
    """
    # Paths to NYT data, tokenized stories, and finished files
    parser.add_argument('--nyt_path', action='store',
                        help='path to data/ in LDC release of NYT corpus')
    parser.add_argument('--tokenized_path', action='store',
                        help='path to store tokenized stories')
    parser.add_argument('--finished_path', action='store',
                        help='path to store finished files')
    
    #Filters for TextRank preprocessing
    parser.add_argument('--textrank', action='store_true', 
                        help='if true, use TextRank preprocessing',
                        default=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NYT summary extraction")
    add_args(parser)
    args = parser.parse_args()

    # Tokenize all .xml files into .story files
    tokenize_all_docs(raw_dir_path=args.nyt_path, tokenize_dir_path=args.tokenized_path,
                      textrank=args.textrank)
    
    # Write .story files to bins, separate train, dev, and test sets, and 
    # create dictionary
    write_to_bin(tokenize_dir_path=args.tokenized_path, finished_files_dir_path=args.finished_path)
    chunk_all(finished_files_dir_path=args.finished_path)
    
