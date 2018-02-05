import src.CONSTANT as CONSTANT


class BaseTagger:
    """
    A base-tagger. Any tagger will inherit this class
    """

    def __do_nothing(self):
        """ Nothing. Just a mangling double underscore test"""
        return "you did something you son of a bitch"

    def __init__(self, filename=None):
        CONSTANT.tagset.extend(('BEGIN', 'END', 'UNKNOWN'))
        self.tagset = CONSTANT.tagset

        self.data = []
        self.vocabulary = {}
        self.N = 0

        self.filename = filename

    def read(self):
        """
        A single line data sample: "Rockwell NNP B-NP"
        From official docs:  The first column contains the current word, the second its part-of-speech tag
        as derived by the Brill tagger and the third its chunk tag as derived from the WSJ corpus
        """
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

    def model(self):
        """
        Save the model.
        :return:
        """
        # TODO
        raise NotImplementedError
