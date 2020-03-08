import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict

assert (np.version.version >= '1.17.0')

from PIL import Image
import time


# from PIL import Image

class gntReader(torch.utils.data.Dataset):
    gnt_head = np.dtype('u4, <u2, u2, u2')

    def __init__(self, paths=[], transform=lambda x: x):
        assert type(paths) == list or type(paths) == str
        self.transform = transform
        self.glyph_to_code = {}
        self.glyph_to_images = defaultdict(list)
        self.code_to_glyph = []
        self.X = []
        self.y = []
        if type(paths) == list:
            for path in paths:
                self.add(path)
        else:
            self.add(paths)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.transform(self.X[index]), self.y[index]

    def add(self, path):
        with open(path, mode='rb') as file:
            while (self._read(file)): pass

    def _read(self, file):
        head_buffer = file.read(self.gnt_head.itemsize)
        if (len(head_buffer) == 0):
            return False
        head = np.frombuffer(head_buffer, dtype=self.gnt_head)
        size, tag, width, height = head[0]
        glyph = tag.tobytes().decode('gb2312')  # gb2312-80

        img = np.frombuffer(file.read(width * height), dtype=np.uint8)
        img = img.reshape(height, width)

        self._add_pair(img, glyph)
        return True

    def _add_pair(self, img, glyph):
        if glyph in self.glyph_to_code:
            code = self.glyph_to_code[glyph]
        else:
            code = np.int64(len(self.code_to_glyph))
            self.code_to_glyph.append(glyph)
            self.glyph_to_code[glyph] = code

        self.glyph_to_images[glyph].append(len(self.X))

        self.X.append(img)
        self.y.append(code)

    def shuffle_and_split(self, ratio, **kwargs):
        indices = list(range(len(self)))
        np.random.shuffle(indices)
        split = round(ratio * len(indices))
        first_indices, second_indices = indices[split:], indices[:split]
        first_sampler = SubsetRandomSampler(first_indices)
        second_sampler = SubsetRandomSampler(second_indices)
        first_loader = torch.utils.data.DataLoader(self, sampler=first_sampler, **kwargs)
        second_loader = torch.utils.data.DataLoader(self, sampler=second_sampler, **kwargs)
        return first_loader, second_loader

