import numpy as np
import torch

assert (np.version.version >= '1.17.0')

from PIL import Image
import time


# from PIL import Image

class gntReader(torch.utils.data.Dataset):
    gnt_head = np.dtype('u4, <u2, u2, u2')

    def __init__(self, paths = [], transform=lambda x: x):
        assert type(paths) == list or type(paths) == str
        self.glyph_to_code = {}
        self.code_to_glyph = []
        self.X = []
        self.y = []
        if type(paths) == list:
            for path in paths:
                self.add(path)
        else:
            self.add(paths)

    def add(self, path):
        with open(path, mode='rb') as file:
            while(self._read(file)): pass


    def _read(self, file):
        head_buffer = file.read(self.gnt_head.itemsize)
        if (len(head_buffer) == 0):
            return False
        head = np.frombuffer(head_buffer, dtype=self.gnt_head)
        size, tag, width, height = head[0]
        glyph = tag.tobytes().decode('gb2312') #gb2312-80

        img = np.frombuffer(file.read(width * height), dtype=np.uint8)
        img = img.reshape(height, width)

        self._add_pair(img, glyph)
        return True

    def _add_pair(self, img, glyph):
        if glyph in self.glyph_to_code:
            code = self.glyph_to_code[glyph]
        else:
            code = len(self.code_to_glyph)
            self.code_to_glyph.append(glyph)
            self.glyph_to_code[glyph] = code

        self.X.append(img)
        self.y.append(code)

    def train_loader(self):

        pass

# gnt_struct = np.dtype([
#     ("size", np.uintc),
#     ("tag", np.byte),
#     ("width", np.ushort),
#     ("height", np.ushort)
# ])
#
# gnt_struct2 = np.dtype([
#     ("size", np.uint32),
#     ("tag", np.uint16),
#     ("width", np.uint16),
#     ("height", np.uint16)
# ])

# prepath = "/Users/iliacherezov/Downloads/HWDB1.1trn_gnt_P1/"
# path = prepath + "1001-c.gnt"
#
# data = np.fromfile(path, dtype=gnt_struct2, count=1)
# print(data)
# print("aligment", gnt_struct2.itemsize)
# pixeles = np.fromfile(path, dtype=np.uint8, offset=10)
# print(len(pixeles))

if __name__ == "__main__":
    reader = gntReader()
    prepath = "/Users/iliacherezov/Downloads/HWDB1.1trn_gnt_P1/"
    #path = prepath + "1001-c.gnt"
    for i in range(3):
        path = prepath + str(1001 + i) + "-c.gnt"
        reader.Read(path)
    # with open(path, mode='rb') as file:  # b is important -> binary
    #     print(type(file))
    #     file.read()
    #     print(len(file.read(5)))

    print(len(reader.y))
    print(len(reader.code_to_glyph))
