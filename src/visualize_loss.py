import torch
from acrnn_model import acRNN
import pytorch_lightning as pl
from data_module import DQMotionData
import sys
import numpy as np


def compare_prediction(prediction, target):
    from matplotlib import pyplot as plt
    diff = prediction.view(target.size()).to(torch.device("cpu")) - target
    diff_np = diff.detach().numpy()

    indices = []
    indices.append(list(range(0, 3)))
    indices.append([i * 8 + x + 3 for i in range(21) for x in range(4)])
    indices.append([i * 8 + x + 7 for i in range(21) for x in range(4)])

    fig, axs = plt.subplots(1, 3, tight_layout=True)
    for i in range(3):
        axs[i].hist(diff_np[:, :, indices[i]].flatten(), bins=100)
    plt.savefig("error_hist.png")

    plt.clf()
    fig, axs = plt.subplots(1, 3, tight_layout=True)
    # across time
    time = diff_np.shape[1]
    instances = diff_np.shape[0]
    diff_divided = np.zeros((3, time))
    for i in range(time):
        for j in range(len(indices)):
            diff_divided[j, i] = np.sum(diff_np[:, i, indices[j]] ** 2) / instances
    for i in range(3):
        axs[i].plot(diff_divided[i, 2:])
    plt.savefig("error_time.png")


def main():
    model = acRNN.load_from_checkpoint(sys.argv[1])
    model.eval()
    data_loader = DQMotionData(sys.argv[2])
    data_loader.setup()
    val_loader = data_loader.val_dataloader()
    for data in val_loader:
        input_data, target = data
        prediction = model((input_data.to(torch.device("cuda:0")), target.to(torch.device("cuda:0"))))
        compare_prediction(prediction, target)
        break


if __name__ == '__main__':
    main()
