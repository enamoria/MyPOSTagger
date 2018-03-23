# Main app scripts will be here
import sys
import os
from src.CONSTANT import tagset, ROOT_DIR
from classes.NaiveTagger import NaiveTagger
from classes.ViterbiDecoder import ViterbiTagger
import time


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


TRAINING_DATA_FILENAME = "train1.txt"
if len(sys.argv) == 2:
    path_to_file = sys.argv[1]
else:
    path_to_file = os.getcwd() + "/data/" + TRAINING_DATA_FILENAME

# test_string = 'These DT B-NP\ninclude VBP B-VP\n, , O\namong IN B-PP\nother JJ B-NP\nparts NNS I-NP\n, , ' \
#               'O\neach DT B-NP\njetliner NN I-NP\n\'s POS B-NP\ntwo CD I-NP\nmajor JJ I-NP\ndiscrimination NN ' \
#               'B-NP\n, , O\na DT B-NP\npressure NN I-NP\nfloor NN I-NP\n, , O\ndiscrimination NN B-NP\nbox NN ' \
#               'I-NP\n, , O\nfixed VBN B-NP\nleading VBG I-NP\nfor IN B-PP\nthe DT B-NP\n ' \
#               'I-NP\nand CC O\nan DT B-NP\nafter JJ I-NP\n. . O '
# tagger = NaiveTagger(test_string, path_to_file)

path = ROOT_DIR + "/data/data_POS_tag_tieng_viet"
filelist = os.listdir(ROOT_DIR + "/data/data_POS_tag_tieng_viet")
print("=======================================")
print("Start ...")

# sum = 0
# for idx, file in enumerate(filelist[:len(filelist) - 3]):
#     os.system("mv " + path + "/" + file + " " + ROOT_DIR + "/data")
#     os.system("mv " + path + "/" + filelist[idx + 1] + " " + ROOT_DIR + "/data")
#     os.system("mv " + path + "/" + filelist[idx + 2] + " " + ROOT_DIR + "/data")
#
#     try:
#         with HiddenPrints():
#             time1 = time.time()
#
#             test_string = open(ROOT_DIR + "/data/" + file).read()
#             test_string += open(ROOT_DIR + "/data/" + filelist[idx + 1]).read()
#             test_string += open(ROOT_DIR + "/data/" + filelist[idx + 2]).read()
#
#             # print(test_string)
#
#             tagger = ViterbiTagger(test_string.replace("\n", " ./. "), datapath="/data/data_POS_tag_tieng_viet")
#             acc = (tagger.tag(test_string.replace("\n", " ./. ")))
#
#         print(file, acc, time.time() - time1)
#         sum += acc
#
#     except Exception as e:
#         print("Error:", e, "occured while testing", file, "continue testing ...")
#
#     os.system("mv " + ROOT_DIR + "/data/" + file + " " + ROOT_DIR + "/data/data_POS_tag_tieng_viet")
#     os.system("mv " + ROOT_DIR + "/data/" + filelist[idx + 1] + " " + ROOT_DIR + "/data/data_POS_tag_tieng_viet")
#     os.system("mv " + ROOT_DIR + "/data/" + filelist[idx + 2] + " " + ROOT_DIR + "/data/data_POS_tag_tieng_viet")
#
# print("==============")
# print("Avg acc:", sum / float(len(filelist)))

test_string = open(ROOT_DIR + "/data/" + "82846.seg.pos").read()
test_string += open(ROOT_DIR + "/data/" + "109075.seg.pos").read()
test_string += open(ROOT_DIR + "/data/" + "88306.seg.pos").read()

tagger = ViterbiTagger(test_string.replace("\n", " ./. "), datapath="/data/data_POS_tag_tieng_viet")
acc = tagger.tag(test_string.replace("\n", " ./. "))

print(acc)

# wordlist = test_string.split("\n")
# wordlist.insert(0, {'word': 'begin', 'chunk-tag': 'BEGIN', 'pos-tag': 'BEGIN'})
# wordlist.insert(len(wordlist), {'word': 'end', 'chunk-tag': 'END', 'pos-tag': 'END'})
#
# tagger = ViterbiTagger(test_string, path_to_file)
# print(tagger.data)
# # xxx = tagger.tag()
# # print(len(xxx))
#
# # print("Result: \n")
# # print("-----------------------------------------")
# # for i in range(1, len(xxx)):
# #     print(wordlist[i-1], tagset[xxx[i]])
# #
