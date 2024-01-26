import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union
import json

from utils.common import PseudoBuffer
from utils.sampling import ConfidenceScaler


class SourceCheckpoint(Callback):
    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):

        checkpoint_filename = (
            "-".join(
                ["source",
                 pl_module.hparams.training_dataset.name,
                 str(trainer.current_epoch)]
            )
            + ".pth"
        )
        os.makedirs(os.path.join(trainer.weights_save_path, 'source_checkpoints'), exist_ok=True)
        checkpoint_path = os.path.join(trainer.weights_save_path, 'source_checkpoints', checkpoint_filename)
        torch.save(pl_module.model.state_dict(), checkpoint_path)


class PseudoCallback(Callback):
    def __init__(self,
                 pseudo_buffer: PseudoBuffer,
                 pseudo_epoch_int: int = 1):
        super().__init__()

        self.pseudo_buffer = pseudo_buffer
        self.pseudo_epoch_int = pseudo_epoch_int

    def on_train_epoch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           unused: Optional = None) -> None:

        epoch = trainer.current_epoch

        if epoch % self.pseudo_epoch_int == 0:
            self.pseudo_buffer.reset_buffer()
            print("!!! BUFFER RESET !!!")

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch

        if epoch % self.pseudo_epoch_int == 0:
            self.pseudo_buffer.check_integrity()
