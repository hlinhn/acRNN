import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_lightning as pl
import numpy as np


class acRNN(pl.LightningModule):
    def __init__(self, in_size=171, hidden_size=1024, out_size=171, cond_num=5, gt_num=5):
        super(acRNN, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.lstm_1 = nn.LSTMCell(self.in_size, self.hidden_size).double()
        self.lstm_2 = nn.LSTMCell(self.hidden_size, self.hidden_size).double()
        self.lstm_3 = nn.LSTMCell(self.hidden_size, self.hidden_size).double()
        self.out = nn.Linear(self.hidden_size, self.out_size).double()

        self.condition_num = cond_num
        self.gt_num = gt_num

    def init_hidden(self, batch):
        cs = []
        hs = []
        for i in range(3):
            cs.append(Variable(torch.DoubleTensor(np.zeros((batch, self.hidden_size))).cuda()))
        for i in range(3):
            hs.append(Variable(torch.DoubleTensor(np.zeros((batch, self.hidden_size))).cuda()))
        return (hs, cs)

    # frame should be stacked (batch, time_steps, input_size)
    def step(self, frame, vec_h, vec_c):
        h0, c0 = self.lstm_1(frame, (vec_h[0], vec_c[0]))
        h1, c1 = self.lstm_2(vec_h[0], (vec_h[1], vec_c[1]))
        h2, c2 = self.lstm_3(vec_h[1], (vec_h[2], vec_c[2]))

        out = self.out(h2)
        new_h = [h0, h1, h2]
        new_c = [c0, c1, c2]

        return (out, new_h, new_c)

    def get_condition_list(self, cond_num, gt_num, seq_len):
        gt_lst = np.ones((100, gt_num))
        cond_lst = np.zeros((100, cond_num))
        lst = np.concatenate((gt_lst, cond_lst), 1).reshape(-1)
        return lst[:seq_len]

    def forward(self, batch_data):
        data, _ = batch_data
        batch = data.size()[0]
        seq_len = data.size()[1]
        cond_list = self.get_condition_list(self.condition_num, self.gt_num, seq_len)
        vec_h, vec_c = self.init_hidden(batch)
        out_seq = Variable(torch.DoubleTensor(np.zeros((batch, 1))).cuda())
        out_frame = Variable(torch.DoubleTensor(np.zeros((batch, self.out_size))).cuda())

        # consider doing something else for teacher forcing here
        for i in range(seq_len):
            if cond_list[i] == 1:
                frame = data[:, i]
            else:
                frame = out_frame
            (out_frame, vec_h, vec_c) = self.step(frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)
        return out_seq[:, 1:out_seq.size()[1]]

    def loss(self, output, batch_data):
        _, target = batch_data
        loss_func = nn.MSELoss()
        loss = loss_func(output, target.view(target.size()[0], -1))
        return loss

    def training_step(self, train_batch, batch_idx):
        predicted = self(train_batch)
        loss = self.loss(predicted, train_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        predicted = self(val_batch)
        loss = self.loss(predicted, val_batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def main():
    model = acRNN()
    model.init_hidden(32)
    print(model.get_condition_list(5, 5, 20))


if __name__ == '__main__':
    main()
