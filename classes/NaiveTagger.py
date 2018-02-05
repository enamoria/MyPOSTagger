import time

from numpy import empty
from src.utils import normalize_input_sentence, evaluate, add_begin_and_trailing_tag
from classes.BaseTagger import BaseTagger
from src.CONSTANT import POS_TAG_KEYNAME, WORD_KEYNAME, TRUETAG_KEYNAME


class NaiveTagger(BaseTagger):
    """
    A naive tagger, using only maximum likelihood estimation.
    No smoothing, unknown word is tagged as unknown
    """

    def __init__(self, sentence=None, filename=None):
        BaseTagger.__init__(self, filename=filename)

        try:
            self.data, self.vocabulary, self.N = self.read()
        except FileNotFoundError:
            print("File not found")

        if sentence is not None:
            self.querySentence = normalize_input_sentence(sentence)
        else:
            print("Query sentence format doesn't supported")

    def probabilities(self, n=3):
        # Argument sentence has to be normalized first

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
            count_t_t = {key: 0 for key in self.tagset}

            # Count c(t_t). For all tags. The function tag_count is for a specific tag
            for i in range(len(self.tagset)):
                if self.tagset[i] in unigram_count_tag:
                    count_t_t[self.tagset[i]] = unigram_count_tag[self.tagset[i]]

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
            self.querySentence.insert(0, {'word': 'begin', 'chunk-tag': 'BEGIN', 'pos-tag': 'BEGIN'})
            self.querySentence.insert(len(self.querySentence), {'word': 'end', 'chunk-tag': 'END', 'pos-tag': 'END'})

        tag_sequence = empty(len(self.querySentence) + 4, dtype=object)

        for i in range(2, len(self.querySentence) - 2):
            start_time = time.time()
            maxprob = 0
            max_index = -1

            print("********************************")
            trial_tag = None
            if self.querySentence[i][WORD_KEYNAME] in self.vocabulary:
                for index in range(len(self.tagset)):
                    trial_tag = self.tagset[index]
                    try:
                        start_time1 = time.time()
                        prob_trial_tag = prob_w_t(self.querySentence[i][WORD_KEYNAME], trial_tag) * \
                                         linear_interpolation_probability(unigram_count_tag, bigram_count_tag,
                                                                          trigram_count_tag, self.querySentence, i,
                                                                          trial_tag)
                        print("prob_w_t execution time ", time.time() - start_time1)
                        if prob_trial_tag > maxprob:
                            maxprob = prob_trial_tag
                            max_index = index
                    except KeyError:
                        pass
                print(self.tagset[max_index] + " " + self.querySentence[i][TRUETAG_KEYNAME] + " " + str(maxprob))
            else:
                print(self.querySentence[i][WORD_KEYNAME] + " can't be found in lexicon. UNKNOWN tag is set")
                max_index = -1

            tag_sequence[i] = self.tagset[max_index]
            self.querySentence[i][POS_TAG_KEYNAME] = self.tagset[max_index]

            print(self.querySentence[i][WORD_KEYNAME] + " execution time: " + str(time.time() - start_time))

        return self.querySentence

    def tag(self):
        return evaluate(self.probabilities())
