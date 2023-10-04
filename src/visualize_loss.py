import torch
from acrnn_model import acRNN
import pytorch_lightning as pl
from data_module import DQMotionData
import sys
import numpy as np
from data_module import plot_data


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


def test_normed(data):
    from matplotlib import pyplot as plt
    data = data.view(data.size(0), 100, -1).cpu().detach().numpy()
    quat_mag = []
    dual_times = []
    for i in range(data.shape[0]):
        for t in range(data.shape[1]):
            for j in range(21):
                q = data[i, t, j * 8 + 3: j * 8 + 7]
                d = data[i, t, j * 8 + 7: j * 8 + 11]
                quat_mag.append(np.sum(q ** 2))
                dual_times.append(np.dot(q, d))
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    axs[0].hist(quat_mag, bins=100)
    axs[1].hist(dual_times, bins=100)
    plt.savefig("normed_dist.png")


def main():
    model = acRNN.load_from_checkpoint(sys.argv[1])
    model.eval()
    data_loader = DQMotionData(sys.argv[2])
    data_loader.setup()
    mean = data_loader.dataset.mean
    std = data_loader.dataset.std
    val_loader = data_loader.val_dataloader()
    for data in val_loader:
        input_data, target = data
        prediction = model((input_data.to(torch.device("cuda:0")), target.to(torch.device("cuda:0")))).cpu().detach()
        compare_prediction(prediction, target)
        random_action = 10
        restored_prediction = prediction.view(prediction.size(0), 100, -1) * std + mean
        restored_target = target * std + mean
        plot_data((restored_prediction, restored_target), idx=random_action)
        test_normed(prediction)
        break


if __name__ == '__main__':
    main()
