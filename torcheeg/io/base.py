from typing import Union

import torch


class _EEGSignalIO:

    @property
    def write_pointer(self):
        return len(self)

    def keys(self):
        raise NotImplementedError

    def eegs(self):
        raise NotImplementedError

    def read_eeg(self, key: str) -> any:
        raise NotImplementedError

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:
        raise NotImplementedError
