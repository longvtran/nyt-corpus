Code to obtain The New York Times Annotated Corpus ([LDC2008T19](https://catalog.ldc.upenn.edu/LDC2008T19)) for summarization. Note: To obtain the corpus, refer to https://catalog.ldc.upenn.edu.

### Citation
This code relies on original scripts written by:

Junyi Jessy Li, Kapil Thadani and Amanda Stent. The Role of Discourse Units in Near-Extractive Summarization. In *Proceedings of the 17th Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL).* 2016.

(Li et al's script is available on Github at: https://github.com/grimpil/nyt-summ)

and partly borrows from scripts on https://github.com/abisee/cnn-dailymail.

The TextRank implementation comes from https://github.com/davidadamojr/TextRank.

### Installation
This script requires NLTK installation:
```
$ pip3 install nltk
```

### Overview
The overall flow of the script is as follows:
  * Read the compress NYT Corpus on disk, conduct preprocessing (notably TextRank), and write into .story files
  * Create chunked of data files in binary format, and split the corpus into train, val, and test set
  * Additionally, a `vocab` file will be created

### Usage
To get started, run:
```
main.py --help
```
