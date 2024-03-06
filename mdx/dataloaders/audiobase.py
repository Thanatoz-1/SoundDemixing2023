import os
import numpy as np
from glob import glob
from pathlib import Path
import random

import audiomentations
import torchaudio as ta

ta.set_audio_backend("soundfile")

import torch
from tqdm import tqdm
from joblib import Parallel, delayed

### ----------------------------------------------------------
# In this piece of code, some useful definitions are
# 1. Sample: Song having individual Instruments and vocals
# 2. Stem: Individual instruments and vocals.
### ----------------------------------------------------------


class AudioBase:
    """Some Information about AudioDataset"""

    def __init__(
        self,
        paths: dict,
        sampling_rate: list,
        targets: list,
        # transforms: audiomentations.Compose, # Make it quick and hardcode them.
        # n_samples: int = 44100,  # This will be a part of the child dataset
        n_channels: int = 2,  # Do you really need 2 channels? Think about this later.
        dtype: str = "PCM_S",
        debug: bool = False,
    ) -> None:
        """
        Base class for audio datasets of all types.
        This class performs basic operations of reading a file and a dataset
        The main objective of the base dataset is to read a file,
        read a dataset and store mapped data

        This class uses torchaudio as backend with soundfile.
        This gives us flexibility to read mp3 as well.

        Args:
            paths: dict = containing the name of source and the list of glob paths
            sampling_rate: list = List of accepted sampling rates
            targets: dict = key, value pair of source and target. The remaining will all be added to "mixtures" key
            n_channels: int = Number of channels
            dtype: str = Type of tensor. This should be either of the available encodings in torchaudio.
                        Read more here: https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html

        """
        super(AudioBase, self).__init__()
        for source in paths:
            if type(paths[source]) != type([]):
                paths[source] = [paths[source]]  # Make sure it is a list.
        self.paths = paths
        self.sampling_rate = sampling_rate
        self.targets = targets
        self.n_channels = n_channels
        self.dtype = dtype
        self.debug = debug
        if self.debug:  # If debug, read only 10 datapoints of each source
            print("In debug mode")
            for source in self.paths:
                self.paths[source] = self.paths[source][:5]

        self.data = {}
        # I was trying to read data as fast as possible but this just doesn't return values into variables.
        # Parallel(n_jobs=-1, verbose=1)(self._read_dataset())

    def _read_dataset(self):
        for source in self.paths.keys():
            for _path in tqdm(self.paths[source], desc=f"Collecting {source}"):
                _sample = os.path.split(os.path.split(_path)[0])[-1]
                if _sample not in self.data:
                    self.data[_sample] = {}  # Create empty dict for string source
                self.data[_sample][source] = self._read_audio(path=_path)
        self.list_keys = list(self.data.keys())

    def _read_audio(self, path: str) -> dict:
        """
        Read the audio files and return the audio channels first.
        """
        info = ta.info(path)
        _sr = info.sample_rate
        _channels = info.num_channels
        _frames = info.num_frames
        assert _sr in self.sampling_rate, f"Found sr = {_sr}. Accepted are [{self.sampling_rate}]"
        audio = self._load_audio(path=path)
        audio.update({"frames": _frames})
        return audio

    def _load_audio(self, path: str, start: int = 0, frames: int = -1):
        ## Returns the wave in (channels, frames)
        # audio = ta.load(
        #     path, frame_offset=start, normalize=True, channels_first=True, format=self.dtype, num_frames=frames
        # )
        audio = ta.load(
            path, frame_offset=start, normalize=True, channels_first=True, num_frames=frames
        )
        return {"wave": audio[0], "sr": audio[1]}

    def _get_mixture(
        self, indexes: int, n_channels: int, n_samples: int, augmentations, applyto_input: bool, applyto_output: bool
    ):
        """Take in the value of index in the form of (source, stem, start_time)
        and return the value if a dictionary form of input and output keys

        Args:
            index (_type_): _description_
        """
        # Create a zero matrix and put the data in the matrix
        ## Create targets and mixtures for the data in the matrix.
        # print(indexes)
        tar = {k: np.zeros(shape=(n_channels, n_samples), order="c", dtype="float32") for k in list(self.paths.keys())}
        mixture = np.zeros(shape=(n_channels, n_samples), order="c", dtype="float32")
        # Take the key from the data and the starting index for the frames.
        for source, stem, start_time in indexes:  # Considering that source will be the key rather than index.
            # print(source, stem, start_time)
            if stem not in list(self.data[source].keys()):  # If the stem does not exists in the source
                continue
            total_samples = self.data[source][stem]["wave"].shape[-1]
            # print("total_samples")
            stem_start_sample = int(np.floor(total_samples * start_time))
            # print("stem_start_sample", stem_start_sample)
            _data = self.data[source][stem]["wave"][:, stem_start_sample : stem_start_sample + n_samples]
            # print("_data", _data.shape)
            padded_data = np.pad(_data, [(0, 0), (0, mixture.shape[1] - _data.shape[1])], mode="constant")
            # print("padded_data", padded_data.shape)
            # print(stem_start_sample, mixture.shape, _data.shape, padded_data.shape)
            # print(_data.min(), _data.max(), padded_data.min(), padded_data.max())
            # print("applying now")
            if np.random.randint(0, 8) % 5 == 0:
                for aug in augmentations.transforms:
                    aug.randomize_parameters(_data, sample_rate=self.sampling_rate[0])
                    # print(aug.__class__.__name__, aug.parameters['should_apply'])

            augmentations.freeze_parameters()
            if applyto_output and applyto_input == False:
                # print("applyto_output", augmentations.transforms[0].parameters)
                mixture += augmentations(padded_data, sample_rate=self.sampling_rate[0])
                tar[stem] += padded_data
            elif applyto_input and applyto_output == False:
                # print("applyto_input", augmentations.transforms[0].parameters)
                mixture += augmentations(padded_data, sample_rate=self.sampling_rate[0])
                tar[stem] += padded_data
            elif applyto_input and applyto_output:
                # print("applyto_input and applyto_output", augmentations.transforms[0].parameters)
                mixture += augmentations(padded_data, sample_rate=self.sampling_rate[0])
                tar[stem] += augmentations(padded_data, sample_rate=self.sampling_rate[0])
            # print("Application done!")
            augmentations.unfreeze_parameters()

        return {"input": mixture, "output": {s: tar[s] for s in tar.keys()}}

    def _get_random_sample(self):
        _stems = list(self.paths.keys())  # select all _stems  ## Also make provision for partial source list
        _sources = [
            list(self.data.keys())[i] for i in np.random.randint(len(self.data.keys()), size=len(_stems))
        ]  # Select random source
        _start_time = [np.random.rand() for i in range(len(_sources))]
        return list(zip(_sources, _stems, _start_time))

    def collate_fn(self, _data) -> None:
        print("Collate function", _data)


class IterableAudioDataset(torch.utils.data.IterableDataset, AudioBase):
    """
    Known issue with generating batches. Please do not use this.
    """

    def __init__(
        self,
        paths: dict,
        augmentations: audiomentations.Compose,
        sampling_rate: list,
        targets: list,
        n_samples: int,
        applyto_input: bool = True,
        applyto_output: bool = True,
        n_channels: int = 2,
        dtype: str = "PCM_S",
        debug: bool = False,
    ) -> None:
        super().__init__(paths, sampling_rate, targets, n_channels, dtype, debug)
        self.augmentations = augmentations
        self.applyto_input = applyto_input
        self.applyto_output = applyto_output
        self.n_samples = n_samples
        # self._read_dataset()

    def __iter__(self):
        samples = self._get_random_sample()
        output = self._get_mixture(
            samples,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            augmentations=self.augmentations,
            applyto_input=self.applyto_input,
            applyto_output=self.applyto_output,
        )
        yield output


class MapAudioDataset(torch.utils.data.Dataset, AudioBase):
    def __init__(
        self,
        paths: dict,
        augmentations: audiomentations.Compose,
        sampling_rate: list,
        targets: list,
        n_samples: int,
        applyto_input: bool = True,
        applyto_output: bool = True,
        n_channels: int = 2,
        dtype: str = "PCM_S",
        debug: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(paths, sampling_rate, targets, n_channels, dtype, debug)
        self.augmentations = augmentations
        self.applyto_input = applyto_input
        self.applyto_output = applyto_output
        self.n_samples = n_samples
        self._read_dataset()

    def __getitem__(self, idx):
        samples = self.list_keys[idx]
        random_samples = [(samples, i, np.random.randn()) for i in self.data[samples].keys()]
        output = self._get_mixture(
            random_samples,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            augmentations=self.augmentations,
            applyto_input=self.applyto_input,
            applyto_output=self.applyto_output,
        )
        return output

    def __len__(self):
        return len(self.data)


class IterableDataset2(torch.utils.data.IterableDataset, AudioBase):
    def __init__(
        self,
        paths: dict,
        augmentations: audiomentations.Compose,
        sampling_rate: list,
        targets: list,
        n_samples: int,
        applyto_input: bool = True,
        applyto_output: bool = True,
        n_channels: int = 2,
        dtype: str = "PCM_S",
        debug: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(paths, sampling_rate, targets, n_channels, dtype, debug)
        self.augmentations = augmentations
        self.n_samples = n_samples
        self.applyto_input = applyto_input
        self.applyto_output = applyto_output

        # self._read_dataset()

    def __iter__(self):
        while True:
            random_samples = self._get_random_sample()
            output = self._get_mixture(
                random_samples,
                n_channels=self.n_channels,
                n_samples=self.n_samples,
                augmentations=self.augmentations,
                applyto_input=self.applyto_input,
                applyto_output=self.applyto_output,
            )
            yield output


if __name__ == "__main__":
    print("Please call individual classes.")
