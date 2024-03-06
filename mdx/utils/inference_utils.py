from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import librosa


def preprocess_audio(audio: np.array, mono: bool, origin_sr: float, sr: float, resample_type: str) -> np.array:
    r"""Preprocess audio to mono / stereo, and resample.

    Args:
        audio: (channels_num, audio_samples), input audio
        mono: bool
        origin_sr: float, original sample rate
        sr: float, target sample rate
        resample_type: str, e.g., 'kaiser_fast'

    Returns:
        output: ndarray, output audio
    """
    if audio.ndim == 1:  # Make it 2d. (channels, samples)
        audio = audio[None, :]
        if not mono:
            audio = np.concatenate([audio, audio], axis=0)
    # If channel first audio comes, then transpose
    if audio.shape[0] > audio.shape[1]:
        audio = audio.T
    if mono:
        audio = np.mean(audio, axis=0)[None, :]
        # (audio_samples,) ## (1, audio_samples)
    if origin_sr != sr:
        output = librosa.core.resample(audio, orig_sr=origin_sr, target_sr=sr, res_type=resample_type)
    else:
        output = audio
    # (audio_samples,) | (channels_num, audio_samples)

    return output


def get_separated_wavs_from_simo_output(x, input_channels, target_source_types):
    r"""Get separated waveforms of target sources from a single input multiple
    output (SIMO) system.

    Args:
        x: (target_sources_num * channels_num, audio_samples)
        input_channels: int
        target_source_types: List[str], e.g., ['vocals', 'bass', ...]

    Returns:
        output_dict: dict, e.g., {
            'vocals': (channels_num, audio_samples),
            'bass': (channels_num, audio_samples),
            ...,
        }
    """
    output_dict = {}

    # for j, source_type in enumerate(target_source_types):
    #     output_dict[source_type] = x[j * input_channels : (j + 1) * input_channels]
    for tar_name, tar_val in zip(target_source_types, np.split(x, len(target_source_types), axis=0)):
        output_dict[tar_name] = tar_val.T

    return output_dict


class Separator:
    def __init__(self, model: nn.Module, segment_samples: int, batch_size: int, device: str):
        r"""Separate to separate an audio clip into a target source.

        Args:
            model: nn.Module, trained model
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
        """
        self.model = model
        self.segment_samples = segment_samples
        self.batch_size = batch_size
        self.device = device

    def separate(self, input_tensor: torch.Tensor) -> np.array:
        r"""Separate an audio clip into a target source.

        Args:
            input_dict: dict, e.g., {
                waveform: (channels_num, audio_samples),
                ...,
            }

        Returns:
            sep_audio: (channels_num, audio_samples) | (target_sources_num, channels_num, audio_samples)
        """
        audio = input_tensor
        audio_samples = audio.shape[-1]

        # Pad the audio with zero in the end so that the length of audio can be
        # evenly divided by segment_samples.
        audio = self.pad_audio(audio)

        # Enframe long audio into segments.
        segments = self.enframe(audio, self.segment_samples)
        # (segments_num, channels_num, segment_samples)

        segments_input_dict = {"waveform": segments}
        # (batch_size, segments_num)

        # Separate in mini-batches.
        sep_segments = self._forward_in_mini_batches(self.model, segments_input_dict, self.batch_size)["waveform"]
        # (segments_num, channels_num, segment_samples)

        # Deframe segments into long audio.
        sep_audio = self.deframe(sep_segments)
        # (channels_num, padded_audio_samples)

        sep_audio = sep_audio[:, 0:audio_samples]
        # (channels_num, audio_samples)

        return sep_audio

    def pad_audio(self, audio: np.array) -> np.array:
        r"""Pad the audio with zero in the end so that the length of audio can
        be evenly divided by segment_samples.

        Args:
            audio: (channels_num, audio_samples)

        Returns:
            padded_audio: (channels_num, audio_samples)
        """
        channels_num, audio_samples = audio.shape

        # Number of segments
        segments_num = int(np.ceil(audio_samples / self.segment_samples))

        pad_samples = segments_num * self.segment_samples - audio_samples

        padded_audio = np.concatenate((audio, np.zeros((channels_num, pad_samples))), axis=1)
        # (channels_num, padded_audio_samples)

        return padded_audio

    def enframe(self, audio: np.array, segment_samples: int) -> np.array:
        r"""Enframe long audio into segments.

        Args:
            audio: (channels_num, audio_samples)
            segment_samples: int

        Returns:
            segments: (segments_num, channels_num, segment_samples)
        """
        audio_samples = audio.shape[1]
        assert audio_samples % segment_samples == 0

        hop_samples = segment_samples // 2
        segments = []

        pointer = 0
        while pointer + segment_samples <= audio_samples:
            segments.append(audio[:, pointer : pointer + segment_samples])
            pointer += hop_samples

        segments = np.array(segments)

        return segments

    def deframe(self, segments: np.array) -> np.array:
        r"""Deframe segments into long audio.

        Args:
            segments: (segments_num, channels_num, segment_samples)

        Returns:
            output: (channels_num, audio_samples)
        """
        if len(segments.shape) == 4:
            (segments_num, _, _, segment_samples) = segments.shape  # For handling 4d cases from MRX and HDemucs
            segments = segments.reshape(segments_num, -1, segment_samples)
        elif len(segments.shape) == 3:
            (segments_num, _, segment_samples) = segments.shape

        if segments_num == 1:
            return segments[0]

        assert self._is_integer(segment_samples * 0.25)
        assert self._is_integer(segment_samples * 0.75)

        output = []

        output.append(segments[0, :, 0 : int(segment_samples * 0.75)])

        for i in range(1, segments_num - 1):
            output.append(segments[i, :, int(segment_samples * 0.25) : int(segment_samples * 0.75)])

        output.append(segments[-1, :, int(segment_samples * 0.25) :])

        output = np.concatenate(output, axis=-1)

        return output

    def _is_integer(self, x: float) -> bool:
        if x - int(x) < 1e-10:
            return True
        else:
            return False

    def _forward_in_mini_batches(self, model: nn.Module, segments_input_dict: Dict, batch_size: int) -> Dict:
        r"""Forward data to model in mini-batch.

        Args:
            model: nn.Module
            segments_input_dict: dict, e.g., {
                'waveform': (segments_num, channels_num, segment_samples),
                ...,
            }
            batch_size: int

        Returns:
            output_dict: dict, e.g. {
                'waveform': (segments_num, channels_num, segment_samples),
            }
        """
        output_dict = {}

        pointer = 0
        segments_num = len(segments_input_dict["waveform"])

        while True:
            if pointer >= segments_num:
                break

            batch_input_dict = {}

            for key in segments_input_dict.keys():
                batch_input_dict[key] = torch.Tensor(segments_input_dict[key][pointer : pointer + batch_size]).to(
                    self.device
                )

            pointer += batch_size

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(batch_input_dict["waveform"])
                if len(batch_input_dict["waveform"]) == 4:
                    batch_output_dict["waveform"] = 0

            # print("Batch output dict shape", batch_output_dict['waveform'].shape)

            for key in batch_output_dict.keys():
                self._append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        return output_dict

    def _append_to_dict(self, dict, key, value) -> None:
        if key in dict.keys():
            dict[key].append(value)
        else:
            dict[key] = [value]
