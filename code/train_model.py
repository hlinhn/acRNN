import torch
from acrnn_model import acRNN
import pytorch_lightning as pl
from data_module import DQMotionData
import sys
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    model = acRNN()
    data_loader = DQMotionData(sys.argv[1])
    logger = TensorBoardLogger("acrnn_logs", name="acrnn")
    trainer = pl.Trainer(max_epochs=1000, logger=logger)
    trainer.fit(model=model, datamodule=data_loader)


if __name__ == '__main__':
    main()
