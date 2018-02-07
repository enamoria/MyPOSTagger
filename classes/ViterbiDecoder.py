import time

from numpy import empty
from src.utils import normalize_input_sentence, evaluate, add_begin_and_trailing_tag, check_for_terminal_argument
from classes.BaseTagger import BaseTagger
from src.CONSTANT import POS_TAG_KEYNAME, WORD_KEYNAME, TRUETAG_KEYNAME, DEFAULT_TRAINING_FILENAME
import sys
import os

class ViterbiTagger(BaseTagger):
    """
    For Decoding: Find the most probable state sequence, given an observation sequence
    This implementation can be used for tagging POS for a sequence
    """

    def __init__(self):
        """
        Constructor
        """
        # TODO Need to seperate input reading into whether a class method or static function

        self.path_to_file = check_for_terminal_argument()
        BaseTagger.__init__(self)

        try:
            self.data, self.vocabulary, self.N = self.read()
        except FileNotFoundError:
            print("File not found")

        if sentence is not None:
            self.querySentence = normalize_input_sentence(sentence)
        else:
            print("Query sentence format doesn't supported")


    def init(self):
        """
        _function_init will init .... and ....
        :return:
        """

