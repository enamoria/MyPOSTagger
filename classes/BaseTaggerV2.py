import os
from src.CONSTANT import POS_TAG_KEYNAME, tagset, ROOT_DIR, RARE_THRESHOLD, RULE_OK_PROBABILITY, punctuations_and_symbols
from src.utils import normalize_input_sentence, n_gram_count_word_joint_tag
from src.rules import *
import numpy as np
from src.IO.reader import Reader
import sys


class BaseTagger:
    """
    A base-tagger. Any tagger will inherit this class
    """

    def __do_nothing(self):
        """ Nothing. Just a mangling double underscore test """
        return "you really did something you son of a bitch"

    def __init__(self, sentence=None, rel_filepath=None, dataset='vi'):
        """
        Constructor
        :param filename: filepath actually #TODO
        """
        self.data = []
        self.vocabulary = {}
        self.N = 0

        self.rel_filepath = rel_filepath

        # The try/catch block below will modify self.tagset
        reader = Reader(rel_filepath)
        try:
            self.data = reader.read(dataset=dataset)

            tagset = []
            vocabulary = {}
            for filedata in self.data:
                for word in filedata:
                    if word[1] not in tagset:
                        tagset.append(word[1])

                    if word[0] not in vocabulary:
                        vocabulary[word[0]] = 1
                    else:
                        vocabulary[word[0]] += 1

            self.vocabulary = vocabulary
            self.rareWords = self.rareWords()
            self.vocabulary['unk'] = len(self.rareWords)

            for filedata in self.data:
                for idx in range(len(filedata)):
                    if filedata[idx][0] in self.rareWords:
                        filedata[idx][0] = 'unk'

            self.tagset = sorted(tagset)

            self.tag_index_dict = self.map_index_to_POS_tag()
            self.vocab_index_dict = self.map_index_to_vocab()

        except FileNotFoundError:
            print(rel_filepath + " not found")
            sys.exit()

        if sentence is not None:
            # self.querySentence = normalize_input_sentence(sentence)
            self.querySentence = [word.split("/") if word != "/" else ['/', '/']
                                  for word in sentence.strip(" ").replace("\n", " ").split(" ")]
        else:
            print("Query sentence format isn't supported")

        ''' Transittion matrix of HMM '''
        self.state_count = self.state_count()
        self.obs_joint_state_count = self.obs_joint_state_count()

        self.ngram_state, self.ngram_obs = self.ngram()

        self.emissions = self.emissions()
        self.transitions = self.transition()

        # print("Start ==================")
        # for i in punctuations_and_symbols:
        #     for j in punctuations_and_symbols:
        #         try:
        #             print(i, j, self.tag_index_dict[i], self.tag_index_dict[j],
        #                   self.emissions[self.tag_index_dict[i]][self.vocab_index_dict[j]])
        #         except KeyError:
        #             print("Nope ...")
        # input("....")
        # print("=========================")
        
        ''' Handling unknown in test set '''
        self.handling_unknown()

    def handling_unknown(self):
        def rule_check(word):
            if isNumber(word):
                return 'M'

            if isNp(word):
                return 'Np'

            if isPunc(word):
                return word

            return None

        def fill_unknown_word_in_test_set():
            unk_word_count = 0
            for word in self.querySentence:
                try:
                    if word[0] not in self.vocab_index_dict:
                        unk_word_count += 1
                        self.vocab_index_dict[word[0]] = len(self.vocab_index_dict)
                        self.vocabulary[word[0]] = 1
                    if word[1] not in self.tag_index_dict:
                        self.tag_index_dict[word[1]] = len(self.tag_index_dict)
                        self.tagset.extend(word[1])

                    j = self.vocab_index_dict[word[0]]
                    # i = self.tag_index_dict[word[1]]

                    check = rule_check(word[0])
                    if check is not None:
                        if word[0] in punctuations_and_symbols:
                            self.emissions[self.tag_index_dict[check]][:] = 0
                            self.emissions[self.tag_index_dict[check]][j] = 1
                        else:
                            self.emissions[self.tag_index_dict[check]][j] = RULE_OK_PROBABILITY

                            # print(check, word)

                        # print(check, self.emissions[self.tag_index_dict[check]][:].tolist().count(1.0))
                # emiss[i][j] = 0.0000000001
                except IndexError:
                    print("Exception:", word)

            print("#unknown word:", unk_word_count)

            # for i in punctuations_and_symbols:
            #     for j in punctuations_and_symbols:
            #         try:
            #             print(i, j, self.tag_index_dict[i], self.tag_index_dict[j], self.emissions[self.tag_index_dict[i]][self.vocab_index_dict[j]])
            #         except KeyError:
            #             print("Nope ...")

        def replace_rare_word_with_unk_training_set():
            for key, value in self.vocabulary.items():
                if value < RARE_THRESHOLD:
                    return

        fill_unknown_word_in_test_set()
        # replace_rare_word_with_unk_training_set()

    def emissions(self):
        """
            Emission matrix. Format {'B': {'Asiad': 0.041666666666666664, 'album': 0.041666666666666664} }
        :return:
        """
        emiss = np.full((len(self.tagset) + 3000, len(self.vocabulary) + 3000), 1/float(len(self.tagset) * len(self.vocabulary)), dtype=np.float64)

        for state in self.obs_joint_state_count:
            i = self.tag_index_dict[state]

            for word in self.obs_joint_state_count[state]:
                j = self.vocab_index_dict[word]

                if state in punctuations_and_symbols and word in punctuations_and_symbols:
                    emiss[i][:] = 0

                emiss[i][j] = self.obs_joint_state_count[state][word] / float(self.state_count[state])

        return emiss

    def transition(self):
        """
            Transition matrix
        :return:
        """
        print("=======================================")
        # transitions = {key[0]: {} for key in self.ngram_state}
        transitions = np.zeros((len(self.tagset) + 300, len(self.tagset) + 300), dtype=np.float64)

        print("Start calculating transition ...")
        for key in self.ngram_state:
            i = self.tag_index_dict[key[0]]
            j = self.tag_index_dict[key[1]]

            tmp = self.ngram_state[key] / float(self.state_count[key[0]])
            transitions[i][j] = tmp
        # return {x[i]}

        print("Done calculating transition!")
        return transitions

    def ngram(self, n=2):
        """
        Calculating ngram with zip():
        :return:
        """
        filelist = sorted(os.listdir(ROOT_DIR + "/" + self.rel_filepath))

        ngram_obs = {}
        ngram_state = {}

        for index, sentence in enumerate(self.data):
            words = []
            states = []

            for idx, word in enumerate(sentence):
                try:
                    if word[0] in self.rareWords:
                        words.append('unk')
                    else:
                        words.append(word[0])
                    states.append(word[1])
                except IndexError:
                    print(index, word, sentence[idx + 1], filelist[index])

            if n == 1:
                tmp_obs_ngram = words
                tmp_state_ngram = states
            elif n == 2:
                tmp_obs_ngram = list(zip(words, words[1:]))
                tmp_state_ngram = list(zip(states, states[1:]))
            elif n == 3:
                tmp_obs_ngram = list(zip(words, words[1:], words[2:]))
                tmp_state_ngram = list(zip(states, states[1:], states[2:]))

            # print(len(tmp_obs_ngram), len(tmp_state_ngram))

            for i in range(len(tmp_obs_ngram)):
                # print(i)
                if tmp_obs_ngram[i] in ngram_obs:
                    ngram_obs[tmp_obs_ngram[i]] += 1
                else:
                    ngram_obs[tmp_obs_ngram[i]] = 1

                if tmp_state_ngram[i] in ngram_state:
                    ngram_state[tmp_state_ngram[i]] += 1
                else:
                    ngram_state[tmp_state_ngram[i]] = 1

        return ngram_state, ngram_obs

    def state_count(self):
        """
            Calculate count(s_i)
        :return: every count(s_i)
        """
        print("=======================================")
        print("Start counting state ...")
        filelist = sorted(os.listdir("/home/enamoria/Dropbox/NLP/POS-tagger/MyTagger/data/data_POS_tag_tieng_viet"))
        state_count = {}
        for idx, sentence in enumerate(self.data):
            for word in sentence:
                if word[1] not in state_count:
                    # print(word[1], filelist[idx])
                    state_count[word[1]] = 1
                else:
                    state_count[word[1]] += 1

        print("Done counting state!")

        return state_count

    def obs_joint_state_count(self):
        """
            Calculate count(o_i, s_j)
        :return: every count(o_i, s_j)
        """

        def find_2nd(string, substring):
            """ There might be word which is sth like //in (tag of characeter '/' We need to split at second '/'"""
            return string.find(substring, string.find(substring) + 1)

        print("=======================================")
        print("Starting counting (o_i, s_j) ...")

        obs_joint_state = {}
        for sentence in self.data:
            for idx in range(len(sentence)):
                # xxx = word.split("/")
                # tag = xxx[1]
                if sentence[idx][1] not in obs_joint_state:  # check for state
                    obs_joint_state[sentence[idx][1]] = {}

                tmp = sentence[idx][0]
                if tmp in self.rareWords:
                    tmp = 'unk'

                if tmp not in obs_joint_state[sentence[idx][1]]:
                    obs_joint_state[sentence[idx][1]][tmp] = 1
                else:
                    obs_joint_state[sentence[idx][1]][tmp] += 1

        print("Done counting obs_joint_state.", len(obs_joint_state), " pairs were found")
        return obs_joint_state

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

    def map_index_to_vocab(self):
        return {tag: mapped_index for mapped_index, tag in enumerate(self.vocabulary.keys())}

    def rareWords(self):
        rare_words = []
        for key, value in self.vocabulary.items():
            if value < RARE_THRESHOLD:
                rare_words.append(key)

        return rare_words

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
                probabilities_obs_given_state[word][tag] = (self.word_tag_joint_count[word][tag]) / (
                    self.POS_count[tag])
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


if __name__ == "__main__":
    tagger = BaseTagger("Toi la cho", "/data/data_POS_tag_tieng_viet")
