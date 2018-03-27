"""Python implementation of the TextRank algoritm.

From this paper:
    https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

Based on:
    https://gist.github.com/voidfiles/1646117
    https://github.com/davidadamojr/TextRank
"""
import editdistance
import io
import itertools
import networkx as nx
import nltk
import os


def setup_environment():
    """Download required resources."""
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print('Completed resource downloads.')


def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    """Apply syntactic filters based on POS tags."""
    return [item for item in tagged if item[1] in tags]


def normalize(tagged):
    """Return a list of tuples with the first item's periods removed."""
    return [(item[0].replace('.', ''), item[1]) for item in tagged]


def unique_everseen(iterable, key=None):
    """List unique elements in order of appearance.

    Examples:
        unique_everseen('AAAABBBCCDAABBB') --> A B C D
        unique_everseen('ABBCcAD', str.lower) --> A B C D
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in [x for x in iterable if x not in seen]:
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def build_graph(nodes):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        levDistance = editdistance.eval(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=levDistance)

    return gr

def extract_key_phrases(text):
    """Return a set of key phrases.

    :param text: A string.
    """
    # tokenize the text using nltk
    word_tokens = nltk.word_tokenize(text)

    # assign POS tags to the words in the text
    tagged = nltk.pos_tag(word_tokens)
    textlist = [x[0] for x in tagged]

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    unique_word_set = unique_everseen([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

    # this will be used to determine adjacent words in order to construct
    # keyphrases with two words

    graph = build_graph(word_set_list)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)

    # the number of keyphrases returned will be relative to the size of the
    # text (a third of the number of vertices)
    one_third = len(word_set_list) // 3
    keyphrases = keyphrases[0:one_third + 1]

    # take keyphrases with multiple words into consideration as done in the
    # paper - if two words are adjacent in the text and are selected as
    # keywords, join them together
    modified_key_phrases = set([])
    # keeps track of individual keywords that have been joined to form a
    # keyphrase
    dealt_with = set([])
    i = 0
    j = 1
    while j < len(textlist):
        first = textlist[i]
        second = textlist[j]
        if first in keyphrases and second in keyphrases:
            keyphrase = first + ' ' + second
            modified_key_phrases.add(keyphrase)
            dealt_with.add(first)
            dealt_with.add(second)
        else:
            if first in keyphrases and first not in dealt_with:
                modified_key_phrases.add(first)

            # if this is the last word in the text, and it is a keyword, it
            # definitely has no chance of being a keyphrase at this point
            if j == len(textlist) - 1 and second in keyphrases and \
                    second not in dealt_with:
                modified_key_phrases.add(second)

        i = i + 1
        j = j + 1

    return modified_key_phrases


def extract_sentences(text, summary_length=100, clean_sentences=False, language='english'):
    """Return a paragraph formatted summary of the source text.

    :param text: A string.
    """
    sent_detector = nltk.data.load('tokenizers/punkt/'+language+'.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    graph = build_graph(sentence_tokens)

    calculated_page_rank = nx.pagerank(graph, weight='weight')
    
    # most important sentences in descending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)

    # return a 100 word summary
    summary = ' '.join(sentences)
    summary_words = summary.split()
    summary_words = summary_words[0:summary_length]
    dot_indices = [idx for idx, word in enumerate(summary_words) if word.find('.') != -1]
    if clean_sentences and dot_indices:
        last_dot = max(dot_indices) + 1
        summary = ' '.join(summary_words[0:last_dot])
    else:
        summary = ' '.join(summary_words)

    return summary

# Return the a dictionary of sentences and their corresponding ranks
def rank_sentences(text, language='english'):
    """Return the sentences and their corresponding ranks

    :param text: A string.
    """
    sent_detector = nltk.data.load('tokenizers/punkt/'+language+'.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    graph = build_graph(sentence_tokens)

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    return calculated_page_rank

# If the text has 4 sentences or less, keep them all. If the text has 5-10 sentences,
# return the 4 most important sentences. If the text has more than 10 sentences, 
# return # sentences / 2 sentences.
def compress_sentences(text, language='english'):
    """Return a specified number of the most important sentences

    :param text: A string.
    """
    sent_detector = nltk.data.load('tokenizers/punkt/'+language+'.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    if len(sentence_tokens) <= 5:
        return text
    elif len(sentence_tokens) > 5 and len(sentence_tokens) <= 10:
        num_keep = 5
    else:
        num_keep = len(sentence_tokens) // 2
    
    graph = build_graph(sentence_tokens)
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    
    # most important sentences in descending order of importance
    ranked_sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)
    removed_sentences = ranked_sentences[num_keep:]
    for key in removed_sentences:
        calculated_page_rank.pop(key, None)
    
    kept_sentences = list(calculated_page_rank.keys())
    kept_text = ' '.join(kept_sentences)

    return kept_text

# If the text has 4 sentences or less, keep them all. If the text has 5-10 sentences,
# return the 4 most important sentences. If the text has more than 10 sentences, 
# return # sentences / 2 sentences.
def compress_sentences_list(sentences_list):
    """Return a specified number of the most important sentences

    :param sentences_list: List of sentences.
    """
    if len(sentences_list) <= 5:
        return sentences_list
    elif len(sentences_list) > 5 and len(sentences_list) <= 10:
        num_keep = 5
    else:
        num_keep = len(sentences_list) // 2
    
    graph = build_graph(sentences_list)
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    
    # most important sentences in descending order of importance
    ranked_sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)
    removed_sentences = ranked_sentences[num_keep:]
    for key in removed_sentences:
        calculated_page_rank.pop(key, None)
    
    kept_sentences_list = list(calculated_page_rank.keys())

    return kept_sentences_list

def write_files(summary, key_phrases, filename):
    """Write key phrases and summaries to a file."""
    print("Generating output to " + 'keywords/' + filename)
    key_phrase_file = io.open('keywords/' + filename, 'w')
    for key_phrase in key_phrases:
        key_phrase_file.write(key_phrase + '\n')
    key_phrase_file.close()

    print("Generating output to " + 'summaries/' + filename)
    summary_file = io.open('summaries/' + filename, 'w')
    summary_file.write(summary)
    summary_file.close()

    print("-")


def summarize_all():
    # retrieve each of the articles
    articles = os.listdir("articles")
    for article in articles:
        print('Reading articles/' + article)
        article_file = io.open('articles/' + article, 'r')
        text = article_file.read()
        keyphrases = extract_key_phrases(text)
        summary = extract_sentences(text)
        write_files(summary, keyphrases, article)

if __name__ == '__main__':
    text = "The challenge put to The Times 's critics was direct : Name a 20th-century work considered `` timeless '' today that you think will be all but forgotten in 100 years ."
    print(rank_sentences(text))
    #print(extract_sentences(text))
    print(compress_sentences(text))