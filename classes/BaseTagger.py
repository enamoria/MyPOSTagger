from src.CONSTANT import POS_TAG_KEYNAME, tagset
from src.utils import normalize_input_sentence, n_gram_count_word_joint_tag
import numpy as np
from src.IO.reader import Reader


class BaseTagger:
    """
    A base-tagger. Any tagger will inherit this class
    """

    def __do_nothing(self):
        """ Nothing. Just a mangling double underscore test """
        return "you really did something you son of a bitch"

    def __init__(self, sentence=None, filename=None):
        """
        Constructor
        :param filename: filepath actually #TODO
        """
        tagset.extend(('BEGIN', 'END', 'UNKNOWN'))
        self.tagset = tagset

        self.data = []
        self.vocabulary = {}
        self.N = 0

        self.filename = filename
        # TODO read file call need to be included here

        # The try/catch block below will modify self.tagset
        try:
            self.data, self.N, self.POS_count = self.read()
            print(self.POS_count)
        except FileNotFoundError:
            print(filename + " not found")

        if sentence is not None:
            self.querySentence = normalize_input_sentence(sentence)
        else:
            print("Query sentence format doesn't supported")

        self.tag_index_dict = self.map_index_to_POS_tag()

        # print(self.POS_count['BEGIN'], self.POS_count['END'])

        """
        Calculate A and B matrices
        Format:
            a:  2-d matrix, a[i][j]: P(s_i -> s_j) (P(s_j | s_i))
            b: P(Word|Tag): dict[word as key][dict of tag]. ex: 'figures': {'NNS': 0.5, 'JJ': 0.2}, 'in': {'IN': 0.2}
        """
        self.HMM_a_matrix = self.transition_probabilities()
        # B is pending ....
        self.word_tag_joint_count = n_gram_count_word_joint_tag(self.data)
        self.HMM_b_matrix = self.emission_probabilities()

        # print("xxx", self.HMM_b_matrix)

    def read(self):
        """
        A single line data sample: "Rockwell NNP B-NP"
        From official docs:  The first column contains the current word, the second its part-of-speech tag
        as derived by the Brill tagger and the third its chunk tag as derived from the WSJ corpus
        """
        POS_count = {tag: 0 for tag in self.tagset}

        f_train = open(self.filename, "r")
        train_sentences = f_train.read().strip("\n").split("\n\n")  # a list which contains sentences

        sentences = []

        word_count = 0

        for sentence in train_sentences:
            if sentence != "":
                words = sentence.split("\n")  # a sublist in train_sentences, which contains words in each sentence
                word_count += len(words)
                tmp_sentence = []

                for word in words:
                    tmp_word = {}
                    POS = word.split(" ")

                    if POS[1] not in self.tagset:
                        self.tagset.append(POS[1])
                        POS_count[POS[1]] = 0

                    POS_count[POS[1]] += 1

                    tmp_word['word'] = POS[0].lower()
                    tmp_word['pos-tag'] = POS[1]
                    tmp_word['chunk-tag'] = POS[2]
                    tmp_sentence.append(tmp_word)

                    if POS[0].lower() not in self.vocabulary:
                        self.vocabulary[POS[0].lower()] = 1
                    else:
                        self.vocabulary[POS[0].lower()] += 1

                # TODO need to be refined: tend to use trigram and bigram, but had to be seperated twice
                tmp_sentence.insert(0, {'word': 'begin', 'chunk-tag': 'BEGIN', 'pos-tag': 'BEGIN'})
                tmp_sentence.insert(len(sentence), {'word': 'end', 'chunk-tag': 'END', 'pos-tag': 'END'})

                sentences.append(tmp_sentence)

        POS_count['BEGIN'] = POS_count['END'] = len(sentences)

        return sentences, word_count, POS_count

    def map_index_to_POS_tag(self):
        """
        This function is to map an index to a specific POS-tag, increasing access ability of the program
        Ex: {'noun':1, 'verb': 2, 'adverb':3, ...}
        This kind of implementation help read through a list of sentences and count
                    POS_i and pair POS_tag without re-iterate the sentence list again :return:
        """
        return {tag: mapped_index for mapped_index, tag in enumerate(self.tagset)}

        # raise NotImplementedError

    def count_POS(self, query_tag):
        """

        :return: transition_matrix
        """
        # return self.HMM_a_matrix[]

        raise NotImplementedError

    def transition_probabilities(self):
        """
        Calculate the transition probabilities (matrix a) of HMM
        :return:
        """
        transition_matrix = np.zeros((len(self.tagset), len(self.tagset)), dtype=np.float16)
        for sentence in self.data:
            for i in range(0, len(sentence) - 1):  # Count from BEGIN tag
                transition_matrix[self.tag_index_dict[sentence[i][POS_TAG_KEYNAME]], self.tag_index_dict[
                    sentence[i + 1][POS_TAG_KEYNAME]]] += 1

        for i in range(np.shape(transition_matrix)[0]):
            for j in range(np.shape(transition_matrix)[1]):
                # a[i][j] mean P(s_i -> s_j)
                if self.POS_count[self.tagset[i]] != 0:
                    transition_matrix[i][j] = (transition_matrix[i][j]) / (float(self.POS_count[self.tagset[i]]))

        return transition_matrix
        # raise NotImplementedError

    def emission_probabilities(self):
        """
        Calculate the emission probabilities (matrix a) of HMM: P(observation | state)
        :return:
        """
        probabilities_obs_given_state = {}
        count = 0
        for word, tags in self.word_tag_joint_count.items():
            probabilities_obs_given_state[word] = {tag: 0.0 for tag in self.tagset}
            for tag in tags:
                probabilities_obs_given_state[word][tag] = (self.word_tag_joint_count[word][tag]) / (self.POS_count[tag])
                # print(probabilities_obs_given_state[word][tag])
                if probabilities_obs_given_state[word][tag] == 0:
                    count += 1

        print("---------------\n", count, "\n--------------------")

        # print(probabilities_obs_given_state)
        return probabilities_obs_given_state

        # raise NotImplementedError

    # PENDING ....
    def n_gram_count_tag(self, n=3):
        """ Count tag in a n-gram model """
        tag_count = {}

        for sentence in self.data:
            for i in range(n - 1):
                sentence.insert(0, {'word': 'begin', 'chunk-tag': 'BEGIN', 'pos-tag': 'BEGIN'})
                sentence.insert(len(sentence), {'word': 'end', 'chunk-tag': 'END', 'pos-tag': 'END'})

            for i in range(n - 1, len(sentence)):
                tmp = ""
                for j in range(i, i - n, -1):
                    tmp = tmp + sentence[j][POS_TAG_KEYNAME] + "\t"

                tmp = tmp[:len(tmp) - 1]
                if tmp not in tag_count:
                    tag_count[tmp] = 1
                else:
                    tag_count[tmp] += 1

        begin = end = ""
        for i in range(n):  # n, not n-1, since in this case we consider the first BEGIN as beginning of the sentence
            begin += "BEGIN\t"
            end += "END\t"
        begin = begin[:len(begin) - 1]
        end = end[:len(end) - 1]

        tag_count[begin] = tag_count[end] = len(self.data)

        return tag_count

    def model(self):
        """
        Save the model.
        :return:
        """
        # TODO
        raise NotImplementedError
