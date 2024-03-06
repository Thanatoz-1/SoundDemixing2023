import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torchlibrosa.stft import STFT

from mdx.utils.pytorch_utils import Base


def l1(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""L1 loss.

    Args:
        output: torch.Tensor
        target: torch.Tensor

    Returns:
        loss: torch.float
    """
    if output.shape != target.shape:
        print("Output inconsistence: ", output.shape, target.shape)
    return torch.mean(torch.abs(output - target))


def l1_wav(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""L1 loss in the time-domain.

    Args:
        output: torch.Tensor
        target: torch.Tensor

    Returns:
        loss: torch.float
    """
    return l1(output, target)


class L1_Wav_L1_Sp(nn.Module, Base):
    def __init__(self):
        r"""L1 loss in the time-domain and L1 loss on the spectrogram."""
        super(L1_Wav_L1_Sp, self).__init__()

        self.window_size = 2048
        hop_size = 441
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""L1 loss in the time-domain and on the spectrogram.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.float
        """

        # L1 loss in the time-domain.
        wav_loss = l1_wav(output, target)

        # L1 loss on the spectrogram.
        sp_loss = l1(
            self.wav_to_spectrogram(output, eps=1e-8),
            self.wav_to_spectrogram(target, eps=1e-8),
        )

        # sp_loss /= math.sqrt(self.window_size)
        # sp_loss *= 1.

        # Total loss.
        return wav_loss + sp_loss

        return sp_loss


class L1_Wav_L1_CompressedSp(nn.Module, Base):
    def __init__(self):
        r"""L1 loss in the time-domain and L1 loss on the spectrogram."""
        super(L1_Wav_L1_CompressedSp, self).__init__()

        self.window_size = 2048
        hop_size = 441
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""L1 loss in the time-domain and on the spectrogram.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.float
        """

        # L1 loss in the time-domain.
        wav_loss = l1_wav(output, target)

        output_mag, output_cos, output_sin = self.wav_to_spectrogram_phase(output, eps=1e-8)
        target_mag, target_cos, target_sin = self.wav_to_spectrogram_phase(target, eps=1e-8)

        mag_loss = l1(output_mag**0.3, target_mag**0.3)
        real_loss = l1(output_mag**0.3 * output_cos, target_mag**0.3 * target_cos)
        imag_loss = l1(output_mag**0.3 * output_sin, target_mag**0.3 * target_sin)

        total_loss = wav_loss + mag_loss + real_loss + imag_loss

        return total_loss


def get_loss_function(loss_type: str) -> Callable:
    r"""Get loss function.

    Args:
        loss_type: str

    Returns:
        loss function: Callable
    """

    if loss_type == "l1_wav":
        return l1_wav

    elif loss_type == "l1_wav_l1_sp":
        return L1_Wav_L1_Sp()

    elif loss_type == "l1_wav_l1_compressed_sp":
        return L1_Wav_L1_CompressedSp()
    elif loss_type == "si_snr":
        return si_snr
    else:
        raise NotImplementedError


EPSILON = 1e-8


def si_snr(estimates: torch.Tensor, targets: torch.Tensor, dim: Optional[int] = -1) -> torch.Tensor:
    """
    Computes the negative scale-invariant signal (source) to noise (distortion) ratio.
    :param estimates (torch.Tensor): estimated source signals, tensor of shape [..., n_samples, ....]
    :param targets (torch.Tensor): ground truth signals, tensor of shape [...., n_samples, ....]
    :param dim (int): time (sample) dimension
    :return (torch.Tensor): estimated SI-SNR with one value for each non-sample dimension
    """
    estimates = _mean_center(estimates, dim=dim)
    targets = _mean_center(targets, dim=dim)
    sig_power = _l2_square(targets, dim=dim, keepdim=True)  # [n_batch, 1, n_srcs]
    dot_ = torch.sum(estimates * targets, dim=dim, keepdim=True)
    scale = dot_ / (sig_power + 1e-12)
    s_target = scale * targets
    e_noise = estimates - s_target
    si_snr_array = _l2_square(s_target, dim=dim) / (_l2_square(e_noise, dim=dim) + EPSILON)
    si_snr_array = -10 * torch.log10(si_snr_array + EPSILON)
    return si_snr_array


def _mean_center(arr: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    mn = torch.mean(arr, dim=dim, keepdim=True)
    return arr - mn


def _l2_square(arr: torch.Tensor, dim: Optional[int] = None, keepdim: Optional[bool] = False) -> torch.Tensor:
    return torch.sum(arr**2, dim=dim, keepdim=keepdim)
