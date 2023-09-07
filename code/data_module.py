import pytorch_lightning as pl
import sys
import numpy as np
from torch.utils import data
import os
import random


class DQDataset(data.Dataset):
    def __init__(self, folder, seq_len):
        self.folder = folder
        self.file_list = self.generate_datalist()
        self.seq_len = seq_len
        self.all_data = [self.convert_to_np(x) for x in self.file_list]
        self.all_data = list(filter(lambda x: x.shape[0] >= seq_len + 20, self.all_data))
        data_len = [x.shape[0] // self.seq_len for x in self.all_data]
        self.all_seq_len = sum(data_len)
        self.index_hash = {}

        curr_ind = 0
        accum_len = np.cumsum(data_len)
        for i in range(self.all_seq_len):
            if i >= accum_len[curr_ind]:
                curr_ind += 1
            self.index_hash[i] = curr_ind

    def generate_datalist(self):
        datafiles = []
        for f in os.listdir(self.folder):
            if not os.path.isdir(os.path.join(self.folder, f)):
                continue
            for npz in os.listdir(os.path.join(self.folder, f)):
                filename = os.path.join(self.folder, f, npz)
                datafiles.append(filename)
        return datafiles

    def convert_to_np(self, filename):
        data = np.load(filename)
        dual = data['arr_0']
        glob = data['arr_1']
        dual = dual.reshape(dual.shape[0], -1)
        composite = np.hstack((glob, dual))
        return composite

    def __len__(self):
        return self.all_seq_len

    # ignoring frame rate for now
    # transpose?
    def __getitem__(self, idx):
        seq_ind = self.index_hash[idx]
        start = random.randint(10, self.all_data[seq_ind].shape[0] - self.seq_len - 10)
        raw_data = self.all_data[seq_ind][start:start + self.seq_len + 1, :]
        trans_diff = raw_data[1:, :3] - raw_data[:raw_data.shape[0] - 1, :3]
        new_data = raw_data[:raw_data.shape[0] - 1, :].copy()
        # predict global displacement, leave height above ground alone
        new_data[:, 0] = trans_diff[:, 0]
        new_data[:, 2] = trans_diff[:, 2]
        return new_data


class DQMotionData(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = DQDataset(data_dir)

    def setup(self, stage):
        # do a random split for now
        pass


# def convert_to
def main():
    test = DQDataset(sys.argv[1], 100)
    # folder = sys.argv[1]
    # motion_data = DQMotionData(sys.argv[1])


if __name__ == '__main__':
    main()
