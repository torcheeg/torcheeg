from typing import Dict, Union

import io
import pywt
import numpy as np
import matplotlib.pyplot as plt

from ..base_transform import EEGTransform


class CWTSpectrum(EEGTransform):
    r'''
        A transform method to convert EEG signals of each channel into spectrograms using wavelet transform.

        .. code-block:: python

            transform = CWTSpectrum()
            transform(eeg=np.random.randn(32, 1000))['eeg'].shape
            >>> (32, 128, 1000)

        Part of the existing work uses :obj:`Resize` to warp the output spectrum to a specified size suitable for CNN processing.

        .. code-block:: python

            transform = Compose([
                CWTSpectrum(),
                ToTensor(),
                Resize([260, 260])
            ])
            transform(eeg=np.random.randn(32, 1000))['eeg'].shape
            >>> (32, 128, 1000)

        When contourf is set to True, a spectrogram of filled contours will be generated for each channel and converted to np.ndarray and returned. This option is usually used for single-channel analysis or visualization of a single channel.

        .. code-block:: python

            transform = CWTSpectrum(contourf=True)
            transform(eeg=np.random.randn(32, 1000))['eeg'].shape
            >>> (32, 480, 640, 4)

        Args:
            sampling_rate (int): The sampling period for the frequencies output in Hz. (default: :obj:`128`)
            wavelet (str): Wavelet to use. Options include: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8, cmor, fbsp, gaus1, gaus2 , gaus3, gaus4, gaus5, gaus6, gaus7, gaus8, mexh, morl, shan. (default: :obj:`'morl'`)
            total_scale: (int): The total wavelet scales to use. (default: :obj:`128`)
            contourf: (bool): Whether to output the np.ndarray corresponding to the image with content of filled contours. (default: :obj:`False`)
            apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
        
        .. automethod:: __call__
    '''
    def __init__(self,
                 sampling_rate: int = 250,
                 wavelet: str = 'morl',
                 total_scale: int = 128,
                 contourf: bool = False,
                 apply_to_baseline: bool = False):
        super(CWTSpectrum, self).__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate
        self.wavelet = wavelet
        self.total_scale = total_scale
        self.contourf = contourf

        fc = pywt.central_frequency(wavelet)
        cparam = 2 * fc * total_scale
        self.scales = cparam / np.arange(1, self.total_scale + 1)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, ...]: The spectrograms based on the wavelet transform for all electrodes. If contourf=False, the output shape is [number of electrodes, total_scale, number of data points]. Otherwise, the output shape is [number of electrodes, height of image, width of image of image, 4], where 4 represents the four channels of the image colors.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
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
    r'''    
        Splitting the EEG signal from each electrode into two functions using wavelet decomposition.

        .. code-block:: python

            transform = DWTDecomposition()
            transform(eeg=np.random.randn(32, 1000))['eeg'].shape
            >>> (32, 500)

        Args:
            apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
        
        .. automethod:: __call__
    '''
    def __init__(self, apply_to_baseline: bool = False):
        super(DWTDecomposition,
              self).__init__(apply_to_baseline=apply_to_baseline)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, 2, number of data points / 2]: EEG signal after wavelet decomposition, where 2 corresponds to the two functions of the wavelet decomposition, and number of data points / 2 represents the length of each component
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return np.stack(pywt.dwt(eeg, 'haar'), axis=0)