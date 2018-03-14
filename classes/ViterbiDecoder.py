import numpy as np

from classes.BaseTagger import BaseTagger
from src.CONSTANT import WORD_KEYNAME
from src.utils import check_for_terminal_argument


class ViterbiTagger(BaseTagger):
    """
    For Decoding: Find the most probable state sequence, given an observation sequence
    This implementation can be used for tagging POS for a sequence
    """

    def __init__(self, sentence=None, filename=None):
        """
        Constructor
        """
        # TODO Need to seperate input reading into whether a class method or static function

        self.path_to_file = check_for_terminal_argument()
        BaseTagger.__init__(self, sentence, filename)

        self.v = []
        self.trace = []

        self.querySentence.insert(0, {'word': 'begin', 'chunk-tag': 'BEGIN', 'pos-tag': 'BEGIN'})
        self.querySentence.insert(len(self.querySentence), {'word': 'end', 'chunk-tag': 'END', 'pos-tag': 'END'})

        # print(self.tagset)

    def init(self):
        """
        _function_init will init v1 and a backpointer (trace, like in dp)
        :return:
        """

        """
        A and B matrices
        Format:
            a:  2-d matrix, a[i][j]: P(s_i -> s_j) (P(s_j | s_i))
            b: P(Word|Tag): dict[word as key][dict of tag]. ex: 'figures': {'NNS': 0.5, 'JJ': 0.2}, 'in': {'IN': 0.2}
        """

        querySentence = self.querySentence

        self.v = np.zeros((len(querySentence) + 1, len(self.tagset)),
                          dtype=np.float64)  # len(querySentence) + 1 because we reserve a spot for "START"
        self.trace = np.zeros((len(querySentence) + 1, len(self.tagset)), dtype=np.int16)

        for index, tag in enumerate(self.tagset):
            " v[i][j] = P(o1, ..., o_i | Q_i = S_j)"

            self.v[1][self.tag_index_dict[tag]] = self.HMM_a_matrix[self.tag_index_dict['BEGIN']][self.tag_index_dict[tag]] * self.HMM_b_matrix[querySentence[1][WORD_KEYNAME]][tag]

            print(tag, self.HMM_a_matrix[self.tag_index_dict['BEGIN']][self.tag_index_dict[tag]], self.HMM_b_matrix[querySentence[1][WORD_KEYNAME]][tag], self.v[1][self.tag_index_dict[tag]])
            self.trace[1][self.tag_index_dict[tag]] = self.tag_index_dict['BEGIN']

    def dp(self):
        """
        Recursion step. Result at step_i will be calculated from step_(i-1)
        :return:
        """
        sentence = self.querySentence
        max_prob = 0
        max_prob_index = 0

        def tracing(trace):
            """
            Trace the respective path for the best probabilities found in self.dp
            Best probabilities is self.v[len(sentence)][\'END\'] ( time = maxtime, tag = terminal tag 'END' )
            :param trace:
            :return: path
            """

            path = []

            i = trace[len(sentence)-1][self.tag_index_dict['END']]
            path.append(i)
            for word_index in reversed(range(len(trace) - 1)):
                i = trace[word_index][i]
                path.append(i)

            return path[::-1]

        # Iterate through the timestamp and states. Here time means len(sentence), counted by words; states means tagset

        for t in range(2, len(sentence)):  # i is time (word)
            # v[t][j] = max(v[t-1][i] * a[i][j] * b[j][o_t])  (i is the iterable)
            for j in range(len(self.tagset)):  # j is state (tag)
                max_prob = 0
                count = 0
                if self.HMM_b_matrix[sentence[t][WORD_KEYNAME]][self.tagset[j]] != 0:
                    for i in range(len(self.tagset)):  # i is state right before j: i assume to be transfered to j at time t
                        tmp = self.v[t - 1][i] * self.HMM_a_matrix[i][j] * self.HMM_b_matrix[sentence[t][WORD_KEYNAME]][
                            self.tagset[j]]

                        count += 1

                        if tmp != 0:
                            print(tmp)

                        if tmp > max_prob:
                            max_prob = tmp
                            max_prob_index = i

                    self.v[t][j] = max_prob
                    self.trace[t][
                        j] = max_prob_index  # Trace: at time t, state j, the state right before j is self.trace[t][j]

        ' In the end, the self.v[len(sentence)][\'END\'] is the best probabilities '

        self.v[len(sentence)-1][self.tag_index_dict['END']] = max([self.v[len(sentence)-1][state_index] \
                                                                 * self.HMM_a_matrix[state_index][
                                                                     self.tag_index_dict['END']] \
                                                                 for state_index in range(len(self.tagset))])
        return tracing(self.trace)

        # raise NotImplementedError

    def tag(self, query_sentence=None):
        """
        Tagger. THis wil perform tagging process for a query sentence
        """
        self.init()
        print("Result is here: ")

        return self.dp()

        # raise NotImplementedError
