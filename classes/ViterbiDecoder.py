import numpy as np

from classes.BaseTaggerV2 import BaseTagger
from src.CONSTANT import WORD_KEYNAME
from src.utils import check_for_terminal_argument


class ViterbiTagger(BaseTagger):
    """
    For Decoding: Find the most probable state sequence, given an observation sequence
    This implementation can be used for tagging POS for a sequence
    """

    def __init__(self, sentence=None, datapath=None):
        """
        Constructor
        """
        # TODO Need to seperate input reading into whether a class method or static function

        self.path_to_file = check_for_terminal_argument()
        BaseTagger.__init__(self, sentence, datapath)

        self.v = []
        self.trace = []

        self.querySentence.insert(0, ['BEGIN', 'BEGIN'])
        self.querySentence.insert(len(self.querySentence), ['END', 'END'])

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

        self.v = np.zeros((len(querySentence), len(self.tagset)),
                          dtype=np.float64)  # len(querySentence) + 1 because we reserve a spot for "START"
        self.trace = np.zeros((len(querySentence), len(self.tagset)), dtype=np.int16)

        for index, tag in enumerate(self.tagset):
            " v[i][j] = P(o1, ..., o_i | Q_i = S_j)"
            # print(self.tag_index_dic
            #
            # f.tag_index_dict[tag]][self.vocab_index_dict[querySentence[1][0]]])

            self.v[1][self.tag_index_dict[tag]] = self.transitions[self.tag_index_dict['BEGIN']][
                                                      self.tag_index_dict[tag]] \
                                                  * self.emissions[self.tag_index_dict[tag]][
                                                      self.vocab_index_dict[querySentence[1][0]]]

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

            i = trace[len(sentence) - 1][self.tag_index_dict['END']]
            path.append(i)
            for word_index in reversed(range(len(trace) - 1)):
                i = trace[word_index][i]
                path.append(i)

            return path[::-1]

        # Iterate through the timestamp and states. Here time means len(sentence), counted by words; states means tagset

        for t in range(2, len(sentence)):  # i is time (word)
            # if t >= 103:
            #     print("t=", t)
            # v[t][j] = max(v[t-1][i] * a[i][j] * b[j][o_t])  (i is the iterable)
            for j in range(len(self.tagset)):  # j is state (tag)
                max_prob = 0
                count = 0
                if self.emissions[self.tag_index_dict[self.tagset[j]]][self.vocab_index_dict[sentence[t][0]]] != 0:
                    for i in range(
                            len(self.tagset)):  # i is state right before j: i assume to be transfered to j at time t
                        tmp = self.v[t - 1][i] * self.transitions[i][j] * \
                              self.emissions[self.tag_index_dict[self.tagset[j]]][self.vocab_index_dict[sentence[t][0]]]

                        # if tmp > 0:
                        #     print(tmp)

                        count += 1

                        if tmp > max_prob:
                            max_prob = tmp
                            max_prob_index = i
                    if max_prob < np.exp(-300):
                        max_prob = max_prob / np.exp(-300)
                    self.v[t][j] = max_prob
                    self.trace[t][
                        j] = max_prob_index  # Trace: at time t, state j, the state right before j is self.trace[t][j]

        ' In the end, the self.v[len(sentence)][\'END\'] is the best probabilities '

        self.v[len(sentence) - 1][self.tag_index_dict['END']] = max([self.v[len(sentence) - 1][state_index] \
                                                                     * self.transitions[state_index][
                                                                         self.tag_index_dict['END']] \
                                                                     for state_index in range(len(self.tagset))])
        return tracing(self.trace)

        # raise NotImplementedError

    def tag(self, query_sentence=None):
        """
        Tagger. THis wil perform tagging process for a query sentence
        """
        self.init()
        print("=======================================")
        print("Result is here: ")

        xxx = self.dp()[1:]
        # for index, x in enumerate(xxx):
        #     print()
        count = 0
        for i, word in enumerate(self.querySentence[:len(self.querySentence) - 1]):
            # print(i, self.tagset[xxx[i]] == word[1], word, "xxx", self.tagset[xxx[i]])
            if self.tagset[xxx[i]] == word[1]:
                count += 1

        # print(count/float(len(self.querySentence)-1))
        return count / float(len(self.querySentence) - 1)

        # raise NotImplementedError
