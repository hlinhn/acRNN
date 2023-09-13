import torch
from acrnn_model import acRNN
import pytorch_lightning as pl
from data_module import DQMotionData
import sys
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback


class PrintCallbacks(Callback):
    def on_train_start(self, trainer, pl_module):
        print(trainer.num_training_batches)


def main():
    model = acRNN()
    data_loader = DQMotionData(sys.argv[1])
    logger = TensorBoardLogger("acrnn_logs", name="acrnn")

    trainer = pl.Trainer(devices=4, max_epochs=5000, logger=logger, strategy="ddp",
                         default_root_dir="/home/halinh/projects/acRNN/logs")
    trainer.fit(model=model, datamodule=data_loader)


if __name__ == '__main__':
    main()
