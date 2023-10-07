#!/usr/bin/env python3

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

from torch.utils.data import Dataset
from VBx.dataset.data_file_loader import DataFileLoader


class BasicDataset(Dataset):
    def __init__(self, file_list, file_loader: DataFileLoader):
        self.file_list = file_list
        self.file_loader = file_loader

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index > len(self.file_list):
            raise IndexError("Index out of range")

        return self.file_loader.load_file(self.file_list[index])
