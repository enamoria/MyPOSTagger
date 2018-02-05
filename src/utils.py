import os
import sys

from src.CONSTANT import DEFAULT_TRAINING_FILENAME


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


def check_for_terminal_argument():
    if len(sys.argv) == 2:
        path_to_file = sys.argv[1]
    else:
        path_to_file = os.getcwd() + "/data/" + DEFAULT_TRAINING_FILENAME

    return path_to_file


# if __name__ == "__main__":
#     test_string = 'Why WRB B-ADVP\nis VBZ O\nthe DT B-NP\nstock NN I-NP\nmarket NN I-NP\nsuddenly RB B-ADVP\nso RB
#     B-ADJP\nvolatile JJ I-ADJP\n? . O '
#
#     print(type(test_string))
#     # test_string = "Rockwell NNP B-NP\n, , O\nbased VBN B-VP\nin IN B-PP\nEl NNP B-NP\nSegundo NNP I-NP\n, , O\nCalif. "
#     #               "NNP B-NP\n, , O\nis VBZ B-VP\nan DT B-NP\naerospace NN I-NP\n, , I-NP\nelectronics NNS I-NP\n, , " \
#     #               "I-NP\nautomotive JJ I-NP\nand CC I-NP\ngraphics NNS I-NP\nconcern VBP I-NP\n. . O "
#     # test_string = "last JJ B-NP\nMay NNP I-NP\n. . O "
#
#     tagger = NaiveTagger(test_string)
#     print(tagger.tag())
#     print(evaluate(tagger.probabilities(normalize_input_sentence(test_string))))
#
#     # print(tagger.tag(open("./data/test.txt", "r")))
#
#     # predicted_sentence = probabilities(normalize_input_sentence(test_string))
#     # print(tagging_accuracy(predicted_sentence))
#
#     # test =