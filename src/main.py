# Main app scripts will be here
import sys
import os
from classes.NaiveTagger import NaiveTagger

TRAINING_DATA_FILENAME = "train.txt"
if len(sys.argv) == 2:
    path_to_file = sys.argv[1]
else:
    path_to_file = os.getcwd() + "/data/" + TRAINING_DATA_FILENAME

test_string = 'These DT B-NP\ninclude VBP B-VP\n, , O\namong IN B-PP\nother JJ B-NP\nparts NNS I-NP\n, , ' \
                  'O\neach DT B-NP\njetliner NN I-NP\n\'s POS B-NP\ntwo CD I-NP\nmajor JJ I-NP\ndiscrimination NN ' \
                  'B-NP\n, , O\na DT B-NP\npressure NN I-NP\nfloor NN I-NP\n, , O\ndiscrimination NN B-NP\nbox NN ' \
                  'I-NP\n, , O\nfixed VBN B-NP\nleading VBG I-NP\nedges NNS I-NP\nfor IN B-PP\nthe DT B-NP\nwings NNS ' \
                  'I-NP\nand CC O\nan DT B-NP\naft JJ I-NP\nkeel NN I-NP\nbeam NN I-NP\n. . O '
tagger = NaiveTagger(test_string, path_to_file)
print(tagger.tag())
