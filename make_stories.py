#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:29:44 2018

@author: longtran
"""

import os
import re
from lexical.splitter import split
from lexical.tokenizer import tokenize
import glob
import tarfile
import logging
from xml.etree import ElementTree
from itertools import chain
from sys import exit
from textrank import compress_sentences_list

from resources import (BAD_DESCRIPTORS, BAD_TITLES, BAD_LEADS,
                       BAD_SUMMLEADS, BAD_SUMMS, BAD_PREFIXES,
                       STITCHES_SUMM, AMBIGUOUS_STITCHES_SUMM,
                       SPLITS_DOC, BAD_ENDWORDS)


# A regex to fix spaces preceding .com in URLs
dotcom_fix_re = re.compile(r"(at +[^ ]+) (?=\.com|\.org|\.net)")

# A regex to identify roundup articles by their lead sentence
roundup_pt_re = re.compile(r"[A-Z\.\- ]+\-\-")

# A regex to identify page markers like [Page A1] or unexpected markers
# like [?][?][?]Author Name
pagemarker_re = re.compile(r" *\[(?:Page )?[A-Z]?[0-9]{1,2}\][\. ]*$")
authmarker_re = re.compile(r" *\[\?\]\[.*")
abstractmarker_re = re.compile(r"\([^()]*\)")

# A regex to identify extraneous periods in online_lead summaries
extraneous_re = re.compile(r"(?:[\?\!]|[\?\!\'\.]\'\')\s*\.$")

# A regex to identify spurious periods in truncated online_lead summaries
incomplete_re = re.compile(r"[\-,:;]\.$")

# A regex to merge hyphenated words separated in online_lead summaries
hyphenated_re = re.compile(r"(?<=[^ \-]\-) (?!and |\(?or |to )")

# A regex to replace single dash tokens in summaries with double dashes
singledash_re = re.compile(r" \- ")

# Regexes to strip prefixes in a single pass, respecting order in the list
prefixes_re = re.compile('^({0})\s*'.format(
                         '|'.join(re.escape(key) for key in BAD_PREFIXES)))

# Regexes to map stitched/split words in a single pass
stitches_summ_re = re.compile(' {0} '.format(
                              ' | '.join(re.escape(key)
                                         for key in STITCHES_SUMM)))
ambi_stitches_re = re.compile(' {0} '.format(
                              ' | '.join(re.escape(key)
                                         for key in AMBIGUOUS_STITCHES_SUMM)))
splits_in_doc_re = re.compile(' {0} '.format(
                              ' | '.join(re.escape(key)
                                         for key in SPLITS_DOC)))

# The abstract sometimes is written in the format: "John Doe reviews ...", where
# John Doe is the name of the article's author. We want to remove the article's
# author name with "Author"
review_markers = ['review', 'reviews', 'comment', 'comments', 'article', 'column']

# A table of Unicode symbol normalizations to match online_lead_paragraph
# summaries to document text
unicode_subs = str.maketrans({'`':       '\'',
                              '´':       '\'',
                              '‘':       '\'',
                              '’':       '\'',
                              '"':       '\'\'',
                              '“':       '\'\'',
                              '”':       '\'\'',
                              '\x86':    '+',
                              '\x91':    '\'',
                              '\x92':    '\'',
                              '\x93':    '\'\'',
                              '\x94':    '\'\'',
                              '\x95':    ' ',
                              '\x96':    '--',
                              '\x97':    '--',
                              '\xa0':    ' ',
                              '\xa9':    '$;',
                              '\xad':    '--',
                              '\xb2':    '2',
                              '\xb7':    '.',
                              '\xbd':    '1/2',
                              '\xbe':    '3/4',
                              u'\u0096': '--',
                              u'\u0097': '--',
                              u'\u2014': '--',
                              u'\u201e': '\'\''})

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

def extract_text(file):
    """Extract UTF-8 text from a file handle.
    """
    lines = []
    for line in file:
        try:
            decoded = line.decode('utf-8', 'strict')
        except UnicodeDecodeError:
            print(line)
            raise
        lines.append(decoded)
    return ''.join(lines) 

def read_block(node, path):
    """Read a block of paragraph-formatted text and return a list
    of paragraph strings.
    """
    paragraphs = []
    for child in node:
        if child.tag != 'p':
            logging.error("Malformed text block in story from {1}"
                          .format(child.tag, path))
            continue
        if child.text is None:
            continue
        paragraphs.append(child.text)
    return paragraphs

def parse_descriptors(node, path):
    """Record descriptors assigned to the document.
    """
    # Tags assigned by the indexing service or automated classifiers
    descriptors = {'indexing': set(),
                        'taxonomic': set(),
                        'online': set(),
                        'online_general': set(),
                        'type': set()}

    for tag_node in node:
        if tag_node.tag != 'classifier':
            # Ignore org, person, book title, etc
            continue

        class_type = (tag_node.attrib['class'], tag_node.attrib['type'])
        label = tag_node.text
        if label is None:
            # Missing labels were observed
            continue
        label = label.title() if label.isupper() else label

        if class_type == ('indexing_service', 'descriptor'):
            descriptors['indexing'].add(label)
        elif class_type == ('online_producer', 'types_of_material'):
            descriptors['type'].add(label)
        elif class_type == ('online_producer', 'taxonomic_classifier'):
            descriptors['taxonomic'].add(label)
        elif class_type == ('online_producer', 'descriptor'):
            descriptors['online'].add(label)
        elif class_type == ('online_producer', 'general_descriptor'):
            descriptors['online_general'].add(label)
        elif class_type not in (('indexing_service', 'names'),
                                ('indexing_service',
                                 'biographical_categories')):
            logging.warning("Unknown classifier '{0!s}' "
                            "in story from {1}"
                            .format(class_type, path))
    return descriptors

def parse_header(node, path):
    """Parse the document header and record descriptors.
    """

    for child in node:
        if child.tag == 'docdata':
            for gchild in child:
                if gchild.tag == 'identified-content':
                    return parse_descriptors(gchild, path)
                elif gchild.tag not in ('doc.copyright', 'series', 'doc-id'):
                    logging.warning("Unknown docdata tag <{0}> "
                                    "in story from {1}"
                                    .format(gchild.tag, path))
        elif child.tag not in ('title', 'meta', 'pubdata'):
            logging.warning("Unknown header tag <{0}> "
                        "in story from {1}"
                            .format(child.tag, path))

def parse_body(node, path):
    # 'normal' and 'online' headlines
    headlines = {}
    story = {}
    for child in node:
        if child.tag == 'body.head':
            for gchild in child:
                if gchild.tag == 'hedline':
                    # Record headlines
                    for ggchild in gchild:
                        if 'class' not in ggchild.attrib:
                            headlines['print'] = ggchild.text
                        elif ggchild.attrib['class'] == 'online_headline':
                            headlines['online'] = ggchild.text
                        else:
                            logging.warning("Unknown headline class {0} "
                                            "in story from {1}".format(
                                                ggchild.attrib['class'],
                                                path))
                if gchild.tag == 'abstract':
                    # Record abstractive summary
                    abstract = read_block(gchild, path)
                    if len(abstract) > 0:
                        story['abstract'] = abstract
        elif child.tag == 'body.content':
            for gchild in child:
                if gchild.attrib['class'] == 'full_text':
                    # Record article text
                    full_text = read_block(gchild, path)
                    if len(full_text) > 0:
                        story['full_text'] = full_text
    return headlines, story

def parse_story(text, path):
    """Parse an XML representation of a story.
    """
    root = ElementTree.fromstring(text)
    story = {}
    descriptors = {}
    
    for node in root:
        if node.tag == 'head':
            descriptors = parse_header(node, path)
        elif node.tag == 'body':
            headlines, story = parse_body(node, path)
        else:
            logging.warning("Unknown top-level tag <{0}> "
                            "in story from {1}"
                            .format(node.tag, path))
    return headlines, story, descriptors

def has_allcaps_summary(abstract):
    """Return whether the summary is all uppercase -- an indication
    that it is a title or location and not a real sentence.
    """
    for paragraph in abstract:
        for sent in paragraph:
            if sent.upper() != sent:
                return False
    return True

def fix_capitalization(tgt_paras, src_paras):
        """Replace uppercase leading words in the target text with equivalent
        mixed-case leading words in the source text.
        """
        tgt = tgt_paras[0]
        src = src_paras[0]
        i = tgt.find(' ')
        if i == -1 or (tgt[-1].isalnum() and tgt.isupper()):
            # Don't edit sentences that appear to be titles
            return tgt_paras

        while tgt[:i].upper() == src[:i].upper():
            if tgt[:i].isupper() or tgt[:i] == tgt[:i].upper():
                j = i + 1 + tgt[i+1:].find(' ')
                if i != j:
                    # Found another space
                    i = j
                    continue
                elif tgt.upper() == src[:len(tgt)].upper():
                    # Replace the whole string if it's identical
                    i = len(tgt)
                else:
                    # Can't find a complete match
                    break

            # Ignore identical spans
            if tgt[:i] == src[:i]:
                break

            # Replace the capitalized prefix and one following word
            logging.warning("Replacing [{0}] -> [{1}] in\n{2}\n"
                            .format(tgt[:i], src[:i], tgt[:80]))
            return [src[:i] + tgt[i:]] + tgt_paras[1:]

        return tgt_paras

def preprocess_full_text(paragraphs):
    """Remove all-caps authors / topics, corrections and page markers.
    Fix spaces before .com.
    """
    processed_paras = []
    allcaps_paras = []

    for p, para in enumerate(paragraphs):
        # Remove all paragraphs following a correction
        if para.startswith('Correction:'):
            break

        # Remove trailing all-caps paragraphs
        if para.upper() == para:
            allcaps_paras.append(para)
            continue

        # Replace page markers
        markers = pagemarker_re.findall(para)
        if len(markers) > 0:
            logging.warning("Dropping page markers {0} from para:\n{1}\n"
                            .format(', '.join(markers), para))
            para = pagemarker_re.sub(' ', para).strip()

        # Replace unknown author markers
        markers = authmarker_re.findall(para)
        if len(markers) > 0:
            logging.warning("Dropping odd marker {0} from summary:\n{1}\n"
                            .format(', '.join(markers), para))
            para = authmarker_re.sub('', para).strip()

        if para != '':
            # Add back all non-trailing all-caps paragraphs
            if len(allcaps_paras) > 0:
                processed_paras.extend(allcaps_paras)
                allcaps_paras = []

            # Fix "nytimes .com" cases
            para = dotcom_fix_re.sub('\\1', para)

            # Add the current paragraph
            processed_paras.append(para)

    return processed_paras

def preprocess_abstract(paragraphs):
    """Normalize Unicode characters. Remove page markers, bureau string
    prefixes and names of subjects. Fix spaces before .com. 
    """
    processed_paras = []

    for p, para in enumerate(paragraphs):
        if p == 0:
            # Remove prefixes from the start of the summary
            new_para = prefixes_re.sub('', para)
            para = new_para.lstrip()
            
            # If the third word or fourth word of the abstract is a review marker word,
            # replace the first two or the first three words of the abstract with 
            # "Author"
            para_split = para.split()
            if len(para_split) >= 3 and para_split[2] in review_markers:
                para_split = ["Author"] + para_split[2:]
            elif len(para_split) >= 4 and para_split[3] in review_markers:
                para_split = ["Author"] + para_split[3:]
            para = ' '.join(para_split)

        # Normalize Unicode characters to match the full text
        para = para.translate(unicode_subs)

        # Replace page markers
        markers = pagemarker_re.findall(para)
        if len(markers) > 0:
            logging.warning("Dropping page marker {0} from summary:\n{1}\n"
                            .format(', '.join(markers), para))
            para = pagemarker_re.sub(' ', para).strip()

        # Replace unknown author markers
        markers = authmarker_re.findall(para)
        if len(markers) > 0:
            logging.warning("Dropping odd marker {0} from summary:\n{1}\n"
                            .format(', '.join(markers), para))
            para = authmarker_re.sub('', para).strip()
        
        # Replace abstract marker at the end of abstracts
        markers = abstractmarker_re.findall(para)
        if len(markers) > 0:
            #logging.warning("Dropping abstract marker {0} from summary:\n{1}\n"
            #                .format(', '.join(markers), para))
            para = para[:len(para)-4] + abstractmarker_re.sub('.', para[len(para)-4:]).strip()

        # Remove extraneous periods from paragraphs. Must follow
        # translation from Unicode symbols.
        if extraneous_re.search(para):
            para = para[:-1]

        if para != '':
            # Fix "nytimes .com" cases
            para = dotcom_fix_re.sub('\\1', para)
            
            # Merge hyphenated words that were incorrectly split into
            # two tokens
            para = hyphenated_re.sub('', para)

            # Expand single dash tokens to double dashes to match the
            # full text
            para = singledash_re.sub(' -- ', para)
        
        # If the abstract ends with multimedia end words (e.g., "photo", "graph"),
        # remove them
        if p == len(paragraphs) - 1:
            para_split = para.split()
            for index, word in reversed(list(enumerate(para_split))):
                if word in BAD_ENDWORDS:
                    del para_split[index]
                else:
                    break
            para = ' '.join(para_split)
        
        # Add the current paragraph
        processed_paras.append(para)
    
    return processed_paras

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."

def preprocess_all(full_text, abstract):
    """After preprocessing, convert each paragraph from a list of paragraphs into a list of
    sentences.
    """
    full_text = fix_capitalization(full_text, abstract)
    full_text = preprocess_full_text(full_text)
    abstract = preprocess_abstract(abstract)
    
    full_text_list = []
    abstract_list = []
    
    for p, paragraph in enumerate(full_text):
        sents_full_text = split(paragraph) if isinstance(paragraph, str) else paragraph
        full_text_list.append(sents_full_text)
    for p, paragraph in enumerate(abstract):
        sents_abstract = split(paragraph) if isinstance(paragraph, str) else paragraph
        abstract_list.append(sents_abstract)
    return full_text_list, abstract_list

def is_templated(headlines, story, descriptors):
    """Return whether this article follows a structure or template that
    makes it inappropriate for the summarization task.
    """
    results = {}
    # Check if the article type descriptors are problematic
    for descriptor in descriptors['type']:
        if descriptor in BAD_DESCRIPTORS:
            results['is_templated'] = True
            return results
            return True

    # Check if the article title indicates a known template
    for title in headlines.values():
        if title in BAD_TITLES:
            results['is_templated'] = True
            return results
            return True

    full_text_list, abstract_list = preprocess_all(story['full_text'], story['abstract'])
    results['full_text_list'] = full_text_list
    results['abstract_list'] = abstract_list
    
    # If preprocessing removes all text in full text or abstract (indicating 
    # that the text is all caps):
    if len(full_text_list) == 0 or len(abstract_list) == 0:
        results['is_templated'] = True
        return results
    
    # Check if the full online lead summary indicates a known template
    if ' '.join(story['abstract']) in BAD_SUMMS:
        results['is_templated'] = True
        return results

    # Check if the first sentence of the online lead summary indicates
    # a known template. Note that this follows preprocessing.
    if abstract_list[0][0] in BAD_SUMMLEADS:
        results['is_templated'] = True
        return results

    # Check if the the first sentence of the article indicates a roundup
    # of sub-stories
    if len(story['full_text']) == 0 or roundup_pt_re.match(story['full_text'][0]):
        results['is_templated'] = True
        return results

    # Check if the first sentence of the article indicates a known
    # template. Note that this follows preprocessing.
    try:
        lead_sent = full_text_list[0][0]
    except IndexError:
        print(full_text_list)
        print(story['full_text'])
        print(story['abstract'])
        exit(1)
        
    if lead_sent in BAD_LEADS:
        results['is_templated'] = True
        return results

    # Check if the first sentence of the article is all uppercase text,
    # often indicating a book review with structured content
    if lead_sent[-1].isalnum() and lead_sent.isupper():
        results['is_templated'] = True
        return results

    results['is_templated'] = False
    return results

def get_tokens(sentences_list):
        """Get just the tokens from an NYT field consisting of a list of
        sentences.
        """
        for sentence in sentences_list:
            for token in tokenize(sentence, warnings=False):
                yield token

def tokenize_all_docs(raw_dir_path, tokenize_dir_path, textrank=False, logger=logging.getLogger(__name__)):
    """Yield stories from disk, optionally filtered through a collection
    of valid stories.
    """
    print("Preparing to tokenize stories in %s to %s..." % (raw_dir_path, tokenize_dir_path))
    if textrank:
        print("Note: TextRank preprocessing is ON...")
    else:
        print("Note: TextRank preprocessing is OFF...")

    years = os.listdir(raw_dir_path)
    i = 0
    for year in years: 
        print("Tokenizing stories from the year %s..." % year)
        month_tarballs = glob.glob(os.path.join(raw_dir_path, year, '*.tgz'))
        # Create a folder to store all stories from a given year
        if not os.path.isdir(tokenize_dir_path):
            os.mkdir(tokenize_dir_path)
        os.mkdir(os.path.join(tokenize_dir_path, year))
            
        for month_tarball in month_tarballs:
            with tarfile.open(month_tarball, 'r:gz') as f:
                for member in f:
                    if not member.isfile():
                        continue

                    # Path of the member inside the tgz files                    
                    path = os.path.join(year, member.name)
                    
                    # Get full name of a story (yyyy_mm_dd_id.story)
                    doc_name = path.replace('/','_').replace('xml', 'story')
                    
                    file = f.extractfile(member)
                    i += 1

                    logging.debug("Reading {0}".format(path))
                    parse_error = False
                    
                    try:
                        contents = extract_text(file)
                        headlines, story, descriptors = parse_story(contents, path)
                        
                    except ElementTree.ParseError:
                        logging.error("Invalid XML in {0}".format(path))
                        parse_error = True
                                        
                    # If the story has full text and abstract, and is parsed correctly
                    if 'full_text' in story and 'abstract' in story and not parse_error:                        
                        # Summary must not be an all-caps title
                        if has_allcaps_summary(story['abstract']):
                            continue
                        
                        # Document must not be a template
                        results_templated = is_templated(headlines, story, descriptors)
                        if results_templated['is_templated']:
                            continue
                        
                        # Abstract must have at least four words
                        if len(results_templated['abstract_list'][0][0].split()) < 4:
                            continue
                        
                        # Convert list of paragraphs to list of sentences
                        abstract = list(chain.from_iterable(results_templated['abstract_list']))
                        full_text = list(chain.from_iterable(results_templated['full_text_list']))
                        
                        # Lower-case everything 
                        abstract = [sentence.lower() for sentence in abstract]
                        full_text = [sentence.lower() for sentence in full_text]
                        
                        # Put periods on the ends of lines that are missing them
                        full_text = [fix_missing_period(sentence) for sentence in full_text]
                        abstract = [fix_missing_period(sentence) for sentence in abstract]
                        
                        # If keep only sentences with the highest TextRank scores:
                        if textrank:
                           full_text = compress_sentences_list(full_text)
                        
                        # Tokenize the full text and abstract
                        full_text = ' '.join(list(get_tokens(full_text)))
                        abstract = ' '.join(list(get_tokens(abstract)))
                        
                        # Join the full text and abstract
                        text = ''.join([full_text, "\n\n", "@abstract", "\n\n", abstract])
                        
                        # Write to .story files
                        with open(os.path.join(tokenize_dir_path, year, doc_name), "w") as tokenized_file:
                            tokenized_file.write(text)
                            tokenized_file.close()
                        
    print("Finish tokenizing all stories!")                        

if __name__ == '__main__':
    hmm = tokenize_all_docs('/home/longtran/Stanford/2017-2018/CS224N/project/data_raw_foo', '/home/longtran/Stanford/2017-2018/CS224N/project/stories_tokenize_foo')