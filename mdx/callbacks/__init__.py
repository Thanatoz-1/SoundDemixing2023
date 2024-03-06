import os
import datetime
import logging
import time
from typing import List

import pickle
import museval
import librosa

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from mdx.utils.inference_utils import Separator, preprocess_audio, get_separated_wavs_from_simo_output


class SaveCheckpointsCallback(pl.Callback):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        checkpoints_dir: str,
        save_step_frequency: int,
    ):
        r"""Callback to save checkpoints every #save_step_frequency steps.

        Args:
            model: nn.Module
            checkpoints_dir: str, directory to save checkpoints
            save_step_frequency: int
        """
        self.model = model
        self.model_name = model_name
        self.checkpoints_dir = checkpoints_dir
        self.save_step_frequency = save_step_frequency
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs):
        r"""Save checkpoint."""
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0:

            checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.model_name}_step={global_step}.pth")

            checkpoint = {"step": global_step, "model": self.model.state_dict(), "metrics": trainer.logged_metrics}

            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Save checkpoint to {checkpoint_path}")


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path
        if os.path.isdir(self.statistics_path):
            self.statistics_path = os.path.join(
                self.statistics_path,
                "statistics.pkl",
            )

        self.backup_statistics_path = "{}_{}.pkl".format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        self.statistics_dict = {"train": [], "test": []}

    def append(self, steps, statistics, split):
        statistics["steps"] = steps
        self.statistics_dict[split].append(statistics)

    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, "wb"))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, "wb"))
        logging.info("    Dump statistics to {}".format(self.statistics_path))
        logging.info("    Dump statistics to {}".format(self.backup_statistics_path))


class Musdb18EvaluationInternalCallback(pl.Callback):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        split_name: str,
        model: nn.Module,
        segment_samples: int,
        batch_size: int,
        device: str,
        evaluate_step_frequency: int,
        # logger: pl.loggers.TensorBoardLogger,
        statistics_container: StatisticsContainer,
    ):
        r"""Callback to evaluate every #save_step_frequency steps for torch dataset.

        Args:
            dataset_dir: str
            model: nn.Module
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
            input_channels: int
            split: 'train' | 'test'
            sample_rate: int
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
            evaluate_step_frequency: int, evaluate every #save_step_frequency steps
            logger: object
            statistics_container: StatisticsContainer
        """
        self.dataset = dataset
        self.model = model
        self.split = split_name
        self.target_source_types = dataset.targets
        self.input_channels = self.dataset.n_channels
        self.sample_rate = self.dataset.sampling_rate[0]
        self.segment_samples = segment_samples
        self.evaluate_step_frequency = evaluate_step_frequency
        # self.logger = logger
        self.statistics_container = statistics_container
        self.mono = dataset.n_channels == 1
        self.resample_type = "kaiser_fast"

        error_msg = "The directory {} is empty!".format(len(self.dataset.data))
        assert len(self.dataset.data) > 0, error_msg

        # separator
        self.separator = Separator(model, self.segment_samples, batch_size, device)

    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs):
        r"""Evaluate separation SDRs of audio recordings."""
        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            sdr_dict = {}

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(self.dataset.data.keys())))

            eval_time = time.time()

            for track in sorted(self.dataset.data.keys())[:5]:  # remove 5 for evaluation

                audio_name = track

                # Get waveform of mixture.
                mixture = torch.stack([i["wave"] for i in self.dataset.data[track].values()]).sum(0)
                # (channels_num, audio_samples)

                mixture = preprocess_audio(
                    audio=mixture,
                    mono=self.mono,
                    origin_sr=self.sample_rate,
                    sr=self.sample_rate,
                    resample_type=self.resample_type,
                )
                # (channels_num, audio_samples)

                target_dict = {}
                sdr_dict[audio_name] = {}

                # Get waveform of all target source types.
                for j, source_type in enumerate(self.target_source_types):
                    # E.g., ['vocals', 'bass', ...]
                    audio = self.dataset.data[track][source_type]["wave"]
                    # (n_channels, samples)

                    audio = preprocess_audio(
                        audio=audio,
                        mono=self.mono,
                        origin_sr=self.sample_rate,
                        sr=self.sample_rate,
                        resample_type=self.resample_type,
                    )
                    # (channels_num, audio_samples)

                    target_dict[source_type] = audio
                    # (channels_num, audio_samples)

                # Separate.
                input_tensor = mixture

                sep_wavs = self.separator.separate(input_tensor)
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                # Post process separation results.
                sep_wavs = preprocess_audio(
                    audio=sep_wavs,
                    mono=self.mono,
                    origin_sr=self.sample_rate,
                    sr=self.sample_rate,
                    resample_type=self.resample_type,
                )
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                sep_wavs = librosa.util.fix_length(sep_wavs, size=mixture.shape[1], axis=1)
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                sep_wav_dict = get_separated_wavs_from_simo_output(
                    sep_wavs, self.input_channels, self.target_source_types
                )
                # output_dict: dict, e.g., {
                #     'vocals': (channels_num, audio_samples),
                #     'bass': (channels_num, audio_samples),
                #     ...,
                # }

                # Evaluate for all target source types.
                for source_type in self.target_source_types:
                    # E.g., ['vocals', 'bass', ...]

                    # Calculate SDR using museval, input shape should be: (nsrc, nsampl, nchan).
                    # print(target_dict[source_type].numpy().shape, sep_wav_dict[source_type].shape)
                    (sdrs, _, _, _) = museval.evaluate(
                        [target_dict[source_type].numpy().T], [sep_wav_dict[source_type]]
                    )

                    sdr = np.nanmedian(sdrs)
                    sdr_dict[audio_name][source_type] = sdr

                    logging.info("{}, {}, sdr: {:.3f}".format(audio_name, source_type, sdr))

            logging.info("-----------------------------")
            median_sdr_dict = {}
            mean_sdr_dict = {}

            # Calculate median SDRs of all songs.
            for source_type in self.target_source_types:
                # E.g., ['vocals', 'bass', ...]

                median_sdr = np.median([sdr_dict[audio_name][source_type] for audio_name in sdr_dict.keys()])
                mean_sdr = np.mean([sdr_dict[audio_name][source_type] for audio_name in sdr_dict.keys()])

                median_sdr_dict[source_type] = median_sdr
                mean_sdr_dict[source_type] = mean_sdr

                logging.info("Step: {}, {}, Median SDR: {:.3f}".format(global_step, source_type, median_sdr))
                logging.info("Step: {}, {}, Mean SDR: {:.3f}".format(global_step, source_type, mean_sdr))

            logging.info("Evaluation time: {:.3f}".format(time.time() - eval_time))

            statistics = {"sdr_dict": sdr_dict, "median_sdr_dict": median_sdr_dict, "mean_sdr": mean_sdr}
            self.statistics_container.append(global_step, statistics, self.split)
            self.statistics_container.dump()
