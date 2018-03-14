import time

from numpy import empty
from src.utils import normalize_input_sentence, evaluate, add_begin_and_trailing_tag, check_for_terminal_argument
from classes.BaseTagger import BaseTagger
from src.CONSTANT import POS_TAG_KEYNAME, WORD_KEYNAME, TRUETAG_KEYNAME, DEFAULT_TRAINING_FILENAME
import sys
import os


# TODO check all document

class ForwardBackward(BaseTagger):
    """
    For Learning: Calculate probability of an observation sequence given a HMM: P(O | lambda)
    """

    def __init__(self):
        """
        Constructor
        """
        # TODO Need to seperate input reading into whether a class method or static function

        self.path_to_file = check_for_terminal_argument()
        BaseTagger.__init__(self)
        raise NotImplementedError

    def probabilities(self):
        """
        Return the probabilities of a hidden state sequence given observed output sequence
        :return:
        """
        raise NotImplementedError

    def prob_given_state(self, start=1, end=len(self.T)):  # , start, end):
        """
        Return the probabilities of output from "start" to "end" given current (hidden) state
        :param start: start of observing time
        :param end: end of observing time
        :return: probabilities.
        ***********************
        *    return format    *
        ***********************
        """

        # for state_index in range(len(self.tagset)):
        #     self.alpha[1][state_index] = 0

        raise NotImplementedError

    def tag(self):
        """
         alpha_t_i: probability of state S[i] at time t with the observed sequence O={o1, ..., oT} with lambda
         model
         """
        self.alpha = self.prob_given_state()

        raise NotImplementedError
