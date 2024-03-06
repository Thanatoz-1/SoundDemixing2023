import h5py
import librosa
import logging

import numpy as np
import torch
from pytorch_lightning.core.datamodule import LightningDataModule


class LightningDataWrapper(LightningDataModule):
    def __init__(self, dataset, val_dataset, batch_size, n_workers):
        r"""Data module.

        Args:
            train_sampler: Sampler object
            train_dataset: Dataset object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.n_workers = n_workers
        self.bs = batch_size
        self.dataset._read_dataset()
        self.val_dataset._read_dataset()

    def setup(self, stage=None):
        r"""called on every device."""
        logging.info(f"Stage is {stage}")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.bs,
            num_workers=self.n_workers,
            pin_memory=True,
        )

        return train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get val loader."""
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.bs,
            num_workers=self.n_workers,
            pin_memory=True,
        )

        return val_loader
