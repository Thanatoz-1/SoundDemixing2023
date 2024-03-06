from typing import Any, Callable, Dict
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio

# from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio as ta


class LitSourceSeparation(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_function: Callable,
        optimizer_type: str,
        learning_rate: float,
        lr_lambda: Callable,
        sr: int,
        vocal_index: int,
        n_channels: int,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            batch_data_preprocessor: object, used for preparing inputs and
                targets for training. E.g., BasicBatchDataPreprocessor is used
                for preparing data in dictionary into tensor.
            model: nn.Module
            loss_function: function
            learning_rate: float
            lr_lambda: function
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "batch_data_preprocessor"])
        self.model = model
        self.optimizer_type = optimizer_type
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda
        self.vocal_index = vocal_index
        self.sr = sr
        self.n_channels = n_channels
        self.stoi = ShortTimeObjectiveIntelligibility(fs=sr, extended=True)
        # self.sdr = ScaleInvariantSignalDistortionRatio(load_diag=0.15, zero_mean=True, use_cg_iter=10)

    def training_step(self, batch_data_dict: Dict, batch_idx: int) -> torch.float:
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. {
                'vocals': (batch_size, channels_num, segment_samples),
                'accompaniment': (batch_size, channels_num, segment_samples),
                'mixture': (batch_size, channels_num, segment_samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # print("Keys in batch data dict: ", batch_data_dict.keys())
        # print("Value in batch data dict: ", batch_data_dict)
        input_dict = batch_data_dict["input"]
        ## debugging input and outputs
        # random_number = np.random.randint(0, 10)
        # logging.info(random_number)
        # if random_number%10==0:
        #     # print(batch_data_dict)
        #     iidx=np.random.randint(0, 10)
        #     ta.save(
        #         f"/mount/arbeitsdaten61/studenten3/advanced-ml/2022/dhyanitr/projects/mdx_train/weights/sample_output/stated_input_{iidx}.wav",
        #         input_dict[0].cpu(),
        #         44100
        #         )
        #     for k,v in batch_data_dict['output'].items():
        #         ta.save(
        #             f"/mount/arbeitsdaten61/studenten3/advanced-ml/2022/dhyanitr/projects/mdx_train/weights/sample_output/stated_{k}_{iidx}.wav",
        #             v[0].cpu(),
        #             44100
        #         )
        #     logging.info("Saved output.")

        targets = torch.concat(list(batch_data_dict["output"].values()), axis=1)

        # input_dict: {
        #     'waveform': (batch_size, channels_num, segment_samples),
        #     (if_exist) 'condition': (batch_size, channels_num),
        # }
        # target_dict: {
        #     'waveform': (batch_size, target_sources_num * channels_num, segment_samples),
        # }

        # Forward.
        self.model.train()

        output_dict = self.model(input_dict)
        # output_dict: {
        #     'waveform': (batch_size, target_sources_num * channels_num, segment_samples),
        # }

        outputs = output_dict["waveform"]
        # outputs:, e.g, (batch_size, target_sources_num * channels_num, segment_samples)

        # Calculate loss.
        loss = self.loss_function(
            output=outputs,
            target=targets,
            mixture=input_dict,
        )
        si = self.vocal_index * self.n_channels
        vocals = outputs[:, si : si + self.n_channels, :]
        inp_vocals = batch_data_dict["output"]["vocals"]
        stoi_score = self.stoi(vocals, inp_vocals)

        total_loss = loss

        self.log_dict({"train/vocal_stoi": stoi_score})
        self.log_dict({"train/loss": loss})

        mean_sdr = scale_invariant_signal_distortion_ratio(outputs, targets).mean()
        self.log_dict({"train/sisdr": mean_sdr})

        return total_loss

    def configure_optimizers(self) -> Any:
        r"""Naive optimizer."""

        if self.optimizer_type == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )

        elif self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )

        else:
            raise NotImplementedError

        scheduler = {
            # "scheduler": LambdaLR(optimizer, self.lr_lambda),
            "scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500000, eta_min=0.00001),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


class CocktailForkModule(pl.LightningModule):
    def __init__(self, model, loss_fn, target_names):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.target_names = target_names

    def _step(self, batch, batch_idx, split):
        input_dict = batch["input"]
        self.model.train()
        targets = torch.stack(list(batch["output"].values()), axis=1)
        y_hat = self.model(input_dict)["waveform"]
        # print("YHAT:", y_hat.shape, targets.shape)
        if y_hat.shape!=targets.shape:
            bs = y_hat.shape[0]
            samples = y_hat.shape[-1]
            targets = targets.view(bs, -1, samples)
        loss = self.loss_fn(y_hat, targets).mean()
        self.log(f"{split}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        x, y, filenames = batch
        y_hat = self.model(x)
        est_sdr = -self.loss_fn(y_hat, y)
        est_sdr = est_sdr.mean(-1).mean(0)  # average of batch and channel
        # expand mixture to shape of isolated sources for noisy SDR
        repeat_shape = len(y.shape) * [1]
        repeat_shape[1] = y.shape[1]
        x = x.unsqueeze(1).repeat(repeat_shape)
        noisy_sdr = -self.loss_fn(x, y)
        noisy_sdr = noisy_sdr.mean(-1).mean(0)  # average of batch and channel
        result_dict = {}
        for i, src in enumerate(self.target_names):
            result_dict[f"noisy_{src}"] = noisy_sdr[i].item()
            result_dict[f"est_{src}"] = est_sdr[i].item()
        self.log_dict(result_dict, on_epoch=True)
        return est_sdr.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        lr_scheduler_config = {"scheduler": lr_scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}
        return [optimizer], [lr_scheduler_config]
