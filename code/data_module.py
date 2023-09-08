import pytorch_lightning as pl
import sys
import numpy as np
from torch.utils.data import Dataset, random_split, Subset, DataLoader
import os
import random
from torchvision import transforms
import torch


class DQDataset(Dataset):
    def __init__(self, folder, seq_len=100):
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

        # subsample to 30FPS
        if "salsa" in filename:
            rate = 2
        else:
            rate = 4
        return composite[::rate, :]

    def __len__(self):
        return self.all_seq_len

    def __getitem__(self, idx):
        seq_ind = self.index_hash[idx]
        start = random.randint(10, self.all_data[seq_ind].shape[0] - self.seq_len - 10)
        raw_data = self.all_data[seq_ind][start:start + self.seq_len + 2, :]
        trans_diff = raw_data[1:, :3] - raw_data[:raw_data.shape[0] - 1, :3]
        new_data = raw_data[:raw_data.shape[0] - 1, :].copy()
        # predict global displacement, leave height above ground alone
        new_data[:, 0] = trans_diff[:, 0]
        new_data[:, 2] = trans_diff[:, 2]
        return torch.from_numpy(new_data[:self.seq_len]).double(), torch.from_numpy(new_data[1:]).double()


class DQMotionData(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = DQDataset(data_dir)
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # do a random split for now
        all_len = self.dataset.all_seq_len
        self.train_idx, self.val_idx = random_split(range(all_len), [int(all_len * 0.7), all_len - int(all_len * 0.7)])

    def train_dataloader(self):
        return DataLoader(Subset(self.dataset, self.train_idx), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(Subset(self.dataset, self.val_idx), batch_size=self.batch_size)


def view_data(data, standard, save_name="test.gif"):
    from visualize import plot_motion
    from position2dual import BVHSkeleton

    skel = BVHSkeleton(standard)
    dual, trans = data
    converted, index = skel.from_dual_to_position(dual, trans)
    plot_motion(converted, index, interval=33, save_path=save_name)


# we want to see the distribution of the numbers from joints
# also want to see if quaternion is normalized
def explore_data(dual, trans):
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    trans = trans.reshape(-1, 3)
    for i in range(3):
        axs[i].hist(trans[:, i])
    plt.savefig("global_trans_hist.png")


def plot_data(data):
    standard_file = "/home/halinh/projects/acRNN/train_data_bvh/standard.bvh"
    in_frame, target = data
    composed = [in_frame[0, :, 3:].numpy(), in_frame[0, :, :3].numpy()]
    composed[0] = composed[0].reshape(composed[0].shape[0], 21, 8)
    view_data(composed, standard_file)
    composed_2 = [target[0, :, 3:].numpy(), target[0, :, :3].numpy()]
    composed_2[0] = composed_2[0].reshape(composed_2[0].shape[0], 21, 8)
    view_data(composed_2, standard_file, "target.gif")


def main():
    test = DQMotionData(sys.argv[1])
    test.setup()

    train_loader = test.train_dataloader()
    for data in train_loader:
        d, _ = data
        dual, trans = d[:, :, 3:].numpy(), d[:, :, :3].numpy()
        explore_data(dual, trans)
        break


if __name__ == '__main__':
    main()
