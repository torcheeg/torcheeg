import io
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pywt

from ..base_transform import EEGTransform


class CWTSpectrum(EEGTransform):
    r"""
        A transform method to convert EEG signals of each channel into spectrograms using wavelet transform.

        .. code-block:: python

            from torcheeg import transforms

            t = transforms.CWTSpectrum()
            t(np.random.randn(32, 1000)).shape
            >>> (32, 128, 1000)

        Part of the existing work uses :obj:`Resize` to warp the output spectrum to a specified size suitable for CNN processing.

        .. code-block:: python

            from torcheeg import transforms

            t = Compose([
                CWTSpectrum(),
                ToTensor(),
                Resize([260, 260])
            ])
            t(np.random.randn(32, 1000)).shape
            >>> (32, 128, 1000)

        When contourf is set to True, a spectrogram of filled contours will be generated for each channel and converted to np.ndarray and returned. This option is usually used for single-channel analysis or visualization of a single channel.

        .. code-block:: python

            from torcheeg import transforms

            t = CWTSpectrum(contourf=True)
            t(np.random.randn(32, 1000)).shape
            >>> (32, 480, 640, 4)

        Args:
            sampling_rate (int): The sampling period for the frequencies output in Hz. (default: :obj:`128`)
            wavelet (str): Wavelet to use. Options include: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8, cmor, fbsp, gaus1, gaus2 , gaus3, gaus4, gaus5, gaus6, gaus7, gaus8, mexh, morl, shan. (default: :obj:`'morl'`)
            total_scale: (int): The total wavelet scales to use. (default: :obj:`128`)
            contourf: (bool): Whether to output the np.ndarray corresponding to the image with content of filled contours. (default: :obj:`False`)


        .. automethod:: __call__
    """

    def __init__(self, sampling_rate: int = 250, wavelet: str = 'morl', total_scale: int = 128, contourf: bool = False):
        super(CWTSpectrum, self).__init__()
        self.sampling_rate = sampling_rate
        self.wavelet = wavelet
        self.total_scale = total_scale
        self.contourf = contourf
        fc = pywt.central_frequency(wavelet)
        cparam = 2 * fc * total_scale
        self.scales = cparam / np.arange(1, self.total_scale + 1)

    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        channel_list = []
        for channel in eeg:
            channel_list.append(self.opt(channel))
        channel_list = np.array(channel_list)
        return np.array(channel_list)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        t = np.arange(0,
                      len(eeg) / self.sampling_rate, 1.0 / self.sampling_rate)
        [cwtmatr, frequencies] = pywt.cwt(eeg, self.scales, self.wavelet,
                                          1.0 / self.sampling_rate)

        if self.contourf:
            fig = plt.figure()

            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1,
                                bottom=0,
                                left=0,
                                right=1,
                                hspace=0,
                                wspace=0)
            plt.margins(0, 0)

            plt.contourf(t, frequencies, abs(cwtmatr))

            with io.BytesIO() as buf:
                fig.savefig(buf, format='raw')
                buf.seek(0)
                img_cwtmatr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                img_cwtmatr = img_cwtmatr.reshape((int(h), int(w), -1))

            return img_cwtmatr

        return cwtmatr

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'wavelet': self.wavelet,
                'total_scale': self.total_scale,
                'contourf': self.contourf
            })


class DWTDecomposition(EEGTransform):
    r"""
        Splitting the EEG signal from each electrode into two functions using wavelet decomposition.

        .. code-block:: python

            from torcheeg import transforms

            t = transforms.DWTDecomposition()
            t(np.random.randn(32, 1000)).shape
            >>> (32, 500)

        Args:


        .. automethod:: __call__
    """

    def __init__(self):
        super(DWTDecomposition, self).__init__()

    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        return np.stack(pywt.dwt(eeg, 'haar'), axis=0)
