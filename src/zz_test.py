import sys

#
# arg0 = sys.argv[0]
# arg1 = sys.argv[1]


# def out(*args):
#     for arg in args:
#         print(arg)
#
#
# print(arg0, " ", arg1)
# print(sys.argv)

# f = open("tagset", "r")
#
# tags = []
# while True:
#     data = f.readline()
#     if data == "":
#         break
#
#     tags.append(data.strip().split("\t")[1])
#
# f.close()
#
# fout = open("CONSTANT.py", "w")
# fout.write("tagset = [")
# for tag in tags:
#     fout.write("\"" + tag + "\"" + ",")
# fout.write("]")
# fout.close()

# temp_dict = {(1, 2, 3): 1, (2, 3, 4):2}
#
# print(temp_dict)
#
# tmp_dict1 = {"NN BEGIN BEGIN\n": 3, "FFF asdF": 4}
# print(tmp_dict1)

import random

# dick = {'1': 22, '2':33, '3':44}
# print(dick)
# for key, value in dick.items():
#     print(key, value)

# for j in range(5,1,-1):
#     print(j)

# x = "XxX"
# y = "YYY"
# print(x.lower())
#
# x = 3
# if x == 2:
#     print("x")
# elif x == 3:
#     print(":y")

# try:
#     x = 1
#     y = x + "sss "
# except TypeError:
#     print("Loi vl")
#
# try:
#     print("????????" + x + 1)
# except Exception:
#     print(" lai loi")

class x():
    def __init__(self):
        self.x = 0
        self._private_x = 1
    def _get(self):
        return self.x

test = x()
print(test.__init__())