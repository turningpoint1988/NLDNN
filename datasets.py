import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data

__all__ = ['SourceDataSet', 'TargetDataSet']


class SourceDataSet(data.Dataset):
    def __init__(self, x, y):
        super(SourceDataSet, self).__init__()
        self.data = x
        self.label = y

        assert len(self.data) == len(self.label), \
            "the number of sequences and labels must be consistent."

        # print("The number of data is {}".format(len(self.label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]

        return {"data": data_one, "label": label_one}


class TargetDataSet(data.Dataset):
    def __init__(self, x):
        super(TargetDataSet, self).__init__()
        self.data = x

        # print("The number of data is {}".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_one = self.data[index]

        return {"data": data_one}
