import os
import sys
import json

os.environ["TORCH_HOME"] = "/tmp/cache"

from pathlib import Path
from glob import glob

import hydra
from omegaconf import DictConfig
import numpy as np
import audiomentations as Au

import torch
from torch.nn import functional as F


from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from mdx.utils import get_logger, get_config_from_path
from mdx.dataloaders.audiobase import MapAudioDataset, IterableDataset2
from mdx.dataloaders import LightningDataWrapper
from mdx.losses import get_loss_function
from mdx.models import LitSourceSeparation, CocktailForkModule
from mdx.callbacks import Musdb18EvaluationInternalCallback, StatisticsContainer, SaveCheckpointsCallback

log = get_logger(__name__)


MAX_WAV_VALUE = 32768.0


def setup_wandb_logger(config):
    os.environ["WANDB_NAME"] = f"{config.name}"
    wandb_logger = WandbLogger(
        name=config.name,
        save_dir=config.workspace,
        offline=config.wandb.offline,
        log_model="all",
        project=config.project,
    )
    wandb_logger.experiment.config.update(config)
    return wandb_logger


def train(config: DictConfig):

    print(config)

    conf = {
        "starting_lr": config.trainer.learning_rate,
        "epochs": config.trainer.epochs,
        "batch_size": config.trainer.bs,
        "seed": config.seed,
        "loss": config.trainer.loss,
        "timesteps": config.timesteps,
        "comment": f"{config.comment}",
    }
    config.device = config.device if config.device != "None" else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device for this project is {config.device}")

    # ================= dataloader main train =============================
    log.info(f"Instantiating dataset <{config.dataset}>")
    paths = get_config_from_path(config.dataset.paths)
    ## Bottleneck code. On hold for integration with hyrda due to inability to call python func in hydra config
    log.info(f"Instantiating dataset <{config.augmentations}>")
    augmentations = hydra.utils.instantiate(config.augmentations)

    dataset = IterableDataset2(
        paths=paths,
        augmentations=augmentations,
        sampling_rate=config.dataloader.sample_rate,
        targets=list(paths.keys()),
        n_samples=config.dataloader.n_samples,
        applyto_input=True,
        applyto_output=True,
        n_channels=config.dataloader.n_channels,
        dtype="PCM_S",
        debug=config.debug,
    )

    # ================= dataloader val train =============================
    val_augmentations = hydra.utils.instantiate(config.augmentations)

    paths = get_config_from_path(config.evaluation.paths)
    val_dataset = MapAudioDataset(
        paths=paths,
        augmentations=val_augmentations,
        sampling_rate=config.dataloader.sample_rate,
        targets=list(paths.keys()),
        n_samples=config.dataloader.n_samples,
        applyto_input=True,
        applyto_output=True,
        n_channels=config.dataloader.n_channels,
        dtype="PCM_S",
        debug=config.debug,
    )

    # lit_val_data = LightningDataWrapper(dataset=val_dataset, n_workers=config.dataloader.n_workers, batch_size=config.dataloader.bs)

    lit_data = LightningDataWrapper(
        dataset=dataset, val_dataset=val_dataset, n_workers=config.dataloader.n_workers, batch_size=config.dataloader.bs
    )
    # ======================================================================

    # =========================== Model ====================================
    log.info(f"Instantiating model <{config.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(config.model)

    if config.model.resume_checkpoint_path:
        checkpoint = torch.load(config.model.resume_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        log.info(f"Load pretrained checkpoint from {config.resume_checkpoint_path}")

    model = model.to(config.device)
    log.info(f"Created model on device {config.device}")

    # =========================== Hyperparams ================================
    loss_function = get_loss_function(loss_type=config.loss)

    ## Needs callbacks here
    save_checkpoints_callback = SaveCheckpointsCallback(
        model=model,
        model_name=config.model.name,
        checkpoints_dir=config.checkpoints_dir,
        save_step_frequency=config.save_step_frequency,
    )

    statistics_container = StatisticsContainer(config.checkpoints_dir)

    evaluate_test_callback = Musdb18EvaluationInternalCallback(
        dataset=val_dataset,
        model=model,
        split_name="test",
        segment_samples=config.segment_samples,
        batch_size=1,
        device=config.device,
        evaluate_step_frequency=config.evaluate_step_frequency,
        statistics_container=statistics_container,
    )
    callbacks = [save_checkpoints_callback, evaluate_test_callback]

    ## Bring in the Lightining Model wrapper and Trainer
    # lit_model = LitSourceSeparation(
    #     model = model,
    #     loss_function=loss_function,
    #     optimizer_type = "Adam",
    #     learning_rate = 1e-3,
    #     lr_lambda = None,
    #     sr = config.dataloader.sample_rate[0],
    #     vocal_index = config.dataloader.n_channels,
    #     n_channels = config.dataloader.n_channels
    # )

    lit_model = CocktailForkModule(model=model, loss_fn=loss_function, target_names=list(paths.keys()))

    trainer = pl.Trainer(
        devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),  # "ddp",
        enable_checkpointing=False,
        callbacks=callbacks,
        max_steps=config.early_stop_steps,
        min_epochs=config.trainer.epochs,
        sync_batchnorm=True,
        precision=config.precision,
        gradient_clip_val=5.0,
        replace_sampler_ddp=False,
        # plugins=[DDPPlugin(find_unused_parameters=False)],
        profiler="simple",
        logger=[],
    )
    trainer.fit(model=lit_model, train_dataloaders=lit_data)
    # model.load_from_checkpoint(config.checkpoints_dir)
    # ckpt = torch.load(checkpoint.best_model_path, map_location="cpu")
    # model_weights = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    # torch.save(model_weights, Path(checkpoint.dirpath) / "best_model.pth")
    # trainer.test(model, lit_val_data)
    log.info(f"Done!")
