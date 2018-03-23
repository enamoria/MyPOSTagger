import os
"""
I dont know if this is significantly necessary or not but I do it for my own comfort
Usual used variables is stored here. Not absolutely CONSTANT though
"""

tagset = ["CC", 'CD', "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS",
          "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT",
          "WP", "WP$", "WRB"]

POS_TAG_KEYNAME = 'pos-tag'
WORD_KEYNAME = 'word'
TRUETAG_KEYNAME = 'true-pos-tag'

dir = os.path.abspath(__file__).split("/")
ROOT_DIR = "/".join(dir[0:len(dir)-2])
print(ROOT_DIR)

DEFAULT_TRAINING_FILENAME = 'train.txt'

# DATASET_SUPPORT = {'vi': self.reader_tieng_viet, 'en': reader_tieng_anh}
# DATASET_SUPPORT = ['vi', 'en']
