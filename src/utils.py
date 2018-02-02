import io
import time
import os
import sys
from numpy import empty
from functools import singledispatch

import src.CONSTANT as CONSTANT

if len(sys.argv) == 2:
    path_to_data = sys.argv[1]
else:
    path_to_data = os.getcwd() + "/data"

POS_TAG_KEYNAME = 'pos-tag'
WORD_KEYNAME = 'word'
TRUETAG_KEYNAME = 'true-pos-tag'
TRAINING_DATA_FILENAME = 'train.txt'

tagset = CONSTANT.tagset


def normalize_input_sentence(raw_sentence):
    words = raw_sentence.split("\n")
    tmp_sentence = []

    for word in words:
        tmp_word = {}
        POS = word.split(" ")

        try:
            if POS[1] not in tagset:
                continue
        except IndexError as e:
            print(e + " " + word)

        # if POS[0] not in vocabulary:
        #     vocabulary[POS[0]] =
        #     continue

        tmp_word['word'] = POS[0].lower()
        tmp_word['pos-tag'] = None
        tmp_word['chunk-tag'] = None
        tmp_word['true-pos-tag'] = POS[1]
        tmp_sentence.append(tmp_word)

    return tmp_sentence


def evaluate(predicted_sentence):
    true_predicted_tag = 0

    for word in predicted_sentence[2:len(predicted_sentence) - 2]:
        if word['pos-tag'] == word['true-pos-tag']:
            true_predicted_tag += 1

    return float(true_predicted_tag) / (len(predicted_sentence) - 4)


def add_begin_and_trailing_tag(words):
    words.insert(0, {'word': 'begin', 'chunk-tag': 'BEGIN', 'pos-tag': 'BEGIN'})
    words.insert(len(words), {'word': 'end', 'chunk-tag': 'END', 'pos-tag': 'END'})


class Tagger:
    def __init__(self):
        tagset.extend(('BEGIN', 'END', 'UNKNOWN'))

        self.data = []
        self.vocabulary = {}
        self.N = 0

        self.data, self.vocabulary, self.N = self.read(TRAINING_DATA_FILENAME)

    def read(self, filename):
        """
        A single line data sample: "Rockwell NNP B-NP"
        From official docs:  The first column contains the current word, the second its part-of-speech tag
        as derived by the Brill tagger and the third its chunk tag as derived from the WSJ corpus
        """
        f_train = open(path_to_data + "/" + filename, "r")
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

                    if POS[1] not in tagset:
                        tagset.append(POS[1])

                    tmp_word['word'] = POS[0].lower()
                    tmp_word['pos-tag'] = POS[1]
                    tmp_word['chunk-tag'] = POS[2]
                    tmp_sentence.append(tmp_word)

                    if POS[0].lower() not in self.vocabulary:
                        self.vocabulary[POS[0].lower()] = 1
                    else:
                        self.vocabulary[POS[0].lower()] += 1

                sentences.append(tmp_sentence)

        return sentences, self.vocabulary, word_count

    def probabilities(self, sentence, n=3):  # Argument sentence has to be normalized first
        def linear_interpolation_probability(unigram, bigram, trigram, sentence, word_index, tag):
            # TODO some thing to optimize lambda
            # For now 3 lambdas are fixed
            """
            :param unigram:
            :param bigram:
            :param trigram:
            :param sentence:
            :param word_index:
            :return: a single number: p(t_i | t_i-1, t_i-2) with linear interpolation
            """
            try:
                trigram_based_prob = trigram[tag + "\t" + sentence[word_index - 1][POS_TAG_KEYNAME] + "\t" +
                                             sentence[word_index - 2][POS_TAG_KEYNAME]] / bigram[
                                         sentence[word_index - 1][POS_TAG_KEYNAME] + "\t" + sentence[word_index - 2][
                                             POS_TAG_KEYNAME]]
            except (KeyError, TypeError):
                trigram_based_prob = 0

            try:
                bigram_based_prob = bigram[tag + "\t" + sentence[word_index - 1][POS_TAG_KEYNAME]] / unigram[
                    sentence[word_index - 1][POS_TAG_KEYNAME]]
            except (KeyError, TypeError):
                bigram_based_prob = 0

            try:
                unigram_based_prob = unigram[tag] / self.N
            except (KeyError, TypeError):
                unigram_based_prob = 0

            return 0.6 * trigram_based_prob + 0.3 * bigram_based_prob + 0.1 * unigram_based_prob

        def n_gram_count_tag(data, n=3):
            """ Count tag in a n-gram model """
            tag_count = {}

            for sentence in data:
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
            for i in range(
                    n):  # n, not n-1, since in this case we consider the first BEGIN as the beginning of the sentence
                begin += "BEGIN\t"
                end += "END\t"
            begin = begin[:len(begin) - 1]
            end = end[:len(end) - 1]

            tag_count[begin] = tag_count[end] = len(data)

            return tag_count

        # prob_t should not be precomputed for every pairs. Only make calculation when needed #
        def prob_t(tags):
            # Note: Delimiter in tags1 and tags2 must be tab character
            # Tags is the numerator, for ex c(t, t-1, t-2) / c(t-1, t-2)

            return trigram_count_tag[tags] / float(bigram_count_tag["\t".join(tags.split("\t")[1:])])

        def tag_count(tag):  # c(t) how many times a tag appeared:
            cnt = 0

            for sentence in self.data:
                for word in sentence:
                    if tag == word[POS_TAG_KEYNAME]:
                        cnt += 1
            return cnt

        def prob_w_t(word, tag):
            """
            Calculate the prob of word given tag
            :param word:
            :param tag:
            :return:
            """
            word = word.lower()
            count_t_t = {key: 0 for key in tagset}

            # Count c(t_t). For all tags. The function tag_count is for a specific tag
            for i in range(len(tagset)):
                if tagset[i] in unigram_count_tag:
                    count_t_t[tagset[i]] = unigram_count_tag[tagset[i]]

            # Coutn c(w_t)
            count_w_t = n_gram_count_word_joint_tag(self.data, n=3)
            # print(tag)

            return count_w_t[word][tag] / float(count_t_t[tag])

        def n_gram_count_word_joint_tag(data, n=3):
            """ Count how many times a word and tag appear together """
            word_tag_count = {}

            def update():
                if word[POS_TAG_KEYNAME] not in word_tag_count[word[WORD_KEYNAME]]:
                    word_tag_count[word[WORD_KEYNAME]][word[POS_TAG_KEYNAME]] = 1
                else:
                    word_tag_count[word[WORD_KEYNAME]][word[POS_TAG_KEYNAME]] += 1

            for sentence in data:
                for word in sentence:
                    if word[WORD_KEYNAME] not in word_tag_count:
                        word_tag_count[word[WORD_KEYNAME]] = {}

                    update()

            return word_tag_count

        unigram_count_tag = n_gram_count_tag(self.data, n=1)
        bigram_count_tag = n_gram_count_tag(self.data, n=2)
        trigram_count_tag = n_gram_count_tag(self.data, n=3)

        for xx in range(n - 1):
            sentence.insert(0, {'word': 'begin', 'chunk-tag': 'BEGIN', 'pos-tag': 'BEGIN'})
            sentence.insert(len(sentence), {'word': 'end', 'chunk-tag': 'END', 'pos-tag': 'END'})

        tag_sequence = empty(len(sentence) + 4, dtype=object)
        # print(unigram_count_tag)
        for i in range(2, len(sentence) - 2):
            start_time = time.time()
            maxprob = 0
            max_index = -1

            print("********************************")
            if sentence[i][WORD_KEYNAME] in self.vocabulary:
                for index, trial_tag in enumerate(tagset):
                    try:
                        prob_trial_tag = prob_w_t(sentence[i][WORD_KEYNAME], trial_tag) * \
                                         linear_interpolation_probability(unigram_count_tag, bigram_count_tag,
                                                                          trigram_count_tag, sentence, i, trial_tag)

                        if prob_trial_tag > maxprob:
                            maxprob = prob_trial_tag
                            max_index = index
                    except KeyError as e:
                        pass
                print(tagset[max_index] + " " + sentence[i][TRUETAG_KEYNAME] + " " + str(maxprob))
            else:
                print(sentence[i][WORD_KEYNAME] + " can't be found in lexicon. UNKNOWN tag is set")
                max_index = -1

            tag_sequence[i] = tagset[max_index]
            sentence[i][POS_TAG_KEYNAME] = tagset[max_index]

            print(sentence[i][WORD_KEYNAME] + " execution time: " + str(time.time() - start_time))

        return sentence

    @singledispatch
    def tag(self, a, b):
        raise NotImplementedError

    @tag.register(str)
    def _(self, a, b):
        """
        Tag a sentence
        :param sentence: The sentence which need to be tag
        :return: Accuracy of tagging
        """
        return a + b

    @tag.register(int)
    def _(self, a, b):
        """
        Tag a sentence
        :param sentence: The sentence which need to be tag
        :return: Accuracy of tagging
        """
        return a + b

    @tag.register(list)
    def _(self, a, b):
        """
        Tag bunch of sentences from a specified file
        :param
        :return: Accuracy of each sentence in the file
        """
        if len(a) == len(b):
            return [a[i] + b[i] for i in range(len(a))]

        return ("DKM")


if __name__ == "__main__":
    # add(1, 2)
    # add('Python', 'Programming')
    # add([1, 2, 3], [5, 6, 7])

    # test_string = 'Why WRB B-ADVP\nis VBZ O\nthe DT B-NP\nstock NN I-NP\nmarket NN I-NP\nsuddenly RB B-ADVP\nso RB
    # B-ADJP\nvolatile JJ I-ADJP\n? . O '
    test_string = 'These DT B-NP\ninclude VBP B-VP\n, , O\namong IN B-PP\nother JJ B-NP\nparts NNS I-NP\n, , ' \
                  'O\neach DT B-NP\njetliner NN I-NP\n\'s POS B-NP\ntwo CD I-NP\nmajor JJ I-NP\ndiscrimination NN ' \
                  'B-NP\n, , O\na DT B-NP\npressure NN I-NP\nfloor NN I-NP\n, , O\ndiscrimination NN B-NP\nbox NN ' \
                  'I-NP\n, , O\nfixed VBN B-NP\nleading VBG I-NP\nedges NNS I-NP\nfor IN B-PP\nthe DT B-NP\nwings NNS ' \
                  'I-NP\nand CC O\nan DT B-NP\naft JJ I-NP\nkeel NN I-NP\nbeam NN I-NP\n. . O '
    # print(type(test_string))
    # # test_string = "Rockwell NNP B-NP\n, , O\nbased VBN B-VP\nin IN B-PP\nEl NNP B-NP\nSegundo NNP I-NP\n, , O\nCalif. " \
    # #               "NNP B-NP\n, , O\nis VBZ B-VP\nan DT B-NP\naerospace NN I-NP\n, , I-NP\nelectronics NNS I-NP\n, , " \
    # #               "I-NP\nautomotive JJ I-NP\nand CC I-NP\ngraphics NNS I-NP\nconcern VBP I-NP\n. . O "
    # # test_string = "last JJ B-NP\nMay NNP I-NP\n. . O "
    #
    tagger = Tagger()
    print(evaluate(tagger.probabilities(normalize_input_sentence(test_string))))

    # # print(tagger.tag(open("./data/test.txt", "r")))
    #
    # # predicted_sentence = probabilities(normalize_input_sentence(test_string))
    # # print(tagging_accuracy(predicted_sentence))
    #
    # # test =
