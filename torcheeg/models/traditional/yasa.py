"""Optional full pretrained YASA sleep-staging workflow, without signal changes."""

import hashlib
import importlib
import math
import numbers
from pathlib import Path
import sys

import numpy as np


# Exact classifiers supplied by the official YASA 0.8.1 wheel on PyPI.
_WEIGHT_HASHES = {
    'clf_eeg+demo_lgb_0.5.0.joblib': '686fd99ad834d657833d7d7c2b3256f9689c023cf13aee16381b2b2cd6f0e86b',
    'clf_eeg+emg+demo_lgb_0.5.0.joblib': '035191c5ffbb84ef9f4b09864bf54f01bd302a94188c34a9d6535460e74f4b49',
    'clf_eeg+emg_lgb_0.5.0.joblib': '9771995bcf7d1c90456986fbf9f991a5b5e8fe68cf0a8c410ec38ad3615a3c78',
    'clf_eeg+eog+demo_lgb_0.5.0.joblib': 'e4762d95dddce707ee8da922dbc505dea4be6a5a5070cbc9a06e0a88f2fc2b43',
    'clf_eeg+eog+emg+demo_lgb_0.5.0.joblib': 'd915de033ecbce0833600151d73a5b271b8f305d63bbc774400ee64bdb9df8ff',
    'clf_eeg+eog+emg_lgb_0.5.0.joblib': '3d27d440c94c61d8013c81c91693ad88810af2ddae85b62f4bc797f6e2464beb',
    'clf_eeg+eog_lgb_0.5.0.joblib': '86d726a3e6aa6195197ffbc9765f5e194c74358295ef331998c21f8badcebcc3',
    'clf_eeg_lgb_0.5.0.joblib': '330c1afdc86396e0b6d9f1294d4c4796beb5f7e7cd38be424c10feae98867e31',
}


class YASASleepStaging:
    r"""Full pretrained YASA 0.8.1 staging with explicit TorchEEG class order.

    This optional traditional baseline runs YASA's complete feature extraction,
    temporal feature context and original pretrained LightGBM classifier. It
    is not an :class:`torch.nn.Module` and has no backpropagation or supervised
    ``fit`` method. Python 3.11+ and the optional ``yasa`` extra are required
    only when predicting. No YASA source or classifier weights are bundled.

    Supply an MNE continuous Raw recording in volts. Do not standardize,
    manually convert to microvolts, filter or independently split 30-second
    epochs before calling this model. YASA handles its own preprocessing and
    full-record feature context. The centered temporal features use future
    context; this is offline inference.

    Args:
        eeg_name (str): Required EEG channel name. Prefer a central derivation
            such as C4-M1 where available.
        eog_name (str, optional): EOG channel name. (default: None)
        emg_name (str, optional): Chin EMG channel name. (default: None)
        metadata (dict, optional): Both ``age`` (years, 0 < age < 120) and
            ``male`` (zero/one or bool), or no demographic features. Only
            provide metadata appropriate to the data and intended protocol.
            (default: None)

    .. code-block:: python

        import mne
        from torcheeg.models import YASASleepStaging

        raw = mne.io.read_raw_edf('recording.edf', preload=True)
        prediction = YASASleepStaging(eeg_name='C4-M1').predict(raw)
        labels = prediction['labels']  # W/N1/N2/N3/REM -> 0/1/2/3/4
        probabilities = prediction['probabilities']  # [epochs, 5]
    """

    BACKEND_VERSION = '0.8.1'
    STAGE_NAMES = ('W', 'N1', 'N2', 'N3', 'REM')
    _BACKEND_STAGES = ('WAKE', 'N1', 'N2', 'N3', 'REM')

    def __init__(self, eeg_name: str, eog_name=None, emg_name=None, metadata=None):
        names = []
        for name, value in [
            ('eeg_name', eeg_name),
            ('eog_name', eog_name),
            ('emg_name', emg_name),
        ]:
            if value is None and name != 'eeg_name':
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(name + ' must be a nonempty channel name.')
            names.append(value)
        if len(names) != len(set(names)):
            raise ValueError('EEG, EOG and EMG names must be distinct.')
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError('metadata must be a dictionary or None.')
        metadata = dict(metadata) if metadata else None
        if metadata is not None:
            if set(metadata) != {'age', 'male'}:
                raise ValueError(
                    'Provide both age and male, or no demographic metadata.'
                )
            age, male = metadata['age'], metadata['male']
            if (
                isinstance(age, bool)
                or not isinstance(age, numbers.Real)
                or not math.isfinite(age)
                or not 0 < age < 120
            ):
                raise ValueError(
                    'age must be finite and strictly between zero and 120 years.'
                )
            if not isinstance(male, (bool, numbers.Integral)) or male not in (0, 1):
                raise ValueError('male must be zero/one or bool.')
            metadata = dict(age=float(age), male=int(male))
        self.eeg_name = eeg_name
        self.eog_name = eog_name
        self.emg_name = emg_name
        self.metadata = metadata

    def _classifier(self, backend):
        name = 'clf_eeg'
        if self.eog_name is not None:
            name += '+eog'
        if self.emg_name is not None:
            name += '+emg'
        if self.metadata is not None:
            name += '+demo'
        name += '_lgb_0.5.0.joblib'
        path = Path(backend.__file__).resolve().parent / 'classifiers' / name
        if not path.is_file():
            raise RuntimeError('Official YASA 0.8.1 classifier missing: ' + name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != _WEIGHT_HASHES[name]:
            raise RuntimeError('Official YASA classifier checksum mismatch: ' + name)
        return path, digest

    def predict(self, raw):
        r"""Predict every complete 30-second epoch in the full continuous input.

        Return labels ``[epochs]`` and probabilities ``[epochs,5]`` in
        :obj:`STAGE_NAMES` order, input-relative ``onsets`` and original-record
        ``recording_onsets`` (seconds), plus protocol/model provenance.
        Discard incomplete trailing epochs as YASA does. No input is changed
        or cached by identity; repeated calls extract fresh features.
        """
        if sys.version_info < (3, 11):
            raise ImportError(
                'YASA 0.8.1 requires Python 3.11+; use an isolated environment and install torcheeg[yasa].'
            )
        try:
            backend = importlib.import_module('yasa')
        except ImportError as error:
            raise ImportError(
                'Install the optional torcheeg[yasa] extra in a Python 3.11+ environment.'
            ) from error
        if getattr(backend, '__version__', None) != self.BACKEND_VERSION:
            raise RuntimeError(
                'This adapter verifies YASA 0.8.1; install torcheeg[yasa] for the matching backend.'
            )
        import mne

        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError('raw must be an MNE continuous Raw recording in volts.')
        names = [
            name
            for name in [self.eeg_name, self.eog_name, self.emg_name]
            if name is not None
        ]
        if any(name not in raw.ch_names for name in names):
            raise ValueError('Every requested EEG/EOG/EMG channel must exist in raw.')
        if raw.info['sfreq'] <= 80:
            raise ValueError(
                'YASA 0.8.1 requires an input sampling frequency greater than 80 Hz.'
            )
        if raw.n_times / raw.info['sfreq'] < 30:
            raise ValueError('At least one complete 30-second epoch is required.')
        # YASA converts these MNE voltage-channel types from volts to microvolts.
        if any(
            kind not in ('eeg', 'eog', 'emg', 'ecg')
            for kind in raw.get_channel_types(picks=names)
        ):
            raise ValueError(
                'Selected channels must have MNE voltage channel types, not misc/stim.'
            )
        if not np.isfinite(raw.get_data(picks=names)).all():
            raise ValueError('Selected voltage channels must contain finite samples.')
        path, digest = self._classifier(backend)
        staging = backend.SleepStaging(
            raw,
            self.eeg_name,
            eog_name=self.eog_name,
            emg_name=self.emg_name,
            metadata=self.metadata,
        )
        # Explicitly use only the hashed classifier from the installed official package.
        hypnogram = staging.predict(path_to_model=str(path))
        stages = hypnogram.hypno.astype(str).to_numpy()
        probabilities = hypnogram.proba
        if set(probabilities.columns) != set(self._BACKEND_STAGES):
            raise RuntimeError('Unexpected YASA probability class set.')
        lookup = {name: i for i, name in enumerate(self._BACKEND_STAGES)}
        if any(stage not in lookup for stage in stages):
            raise RuntimeError('Unexpected YASA predicted sleep stage.')
        labels = np.asarray([lookup[stage] for stage in stages], dtype=np.int64)
        proba = probabilities.loc[:, list(self._BACKEND_STAGES)].to_numpy(
            dtype=np.float64, copy=True
        )
        count = staging.data.shape[-1] // int(30 * staging.sf)
        if (
            proba.shape != (count, 5)
            or labels.shape != (count,)
            or not np.isfinite(proba).all()
            or np.any(proba < 0)
            or not np.allclose(proba.sum(1), 1, rtol=0, atol=1e-6)
        ):
            raise RuntimeError('Invalid full-record YASA probabilities or epoch count.')
        onsets = np.arange(count, dtype=np.float64) * 30
        provenance = dict(
            backend='YASA',
            backend_version=self.BACKEND_VERSION,
            classifier_file=path.name,
            classifier_sha256=digest,
            stage_names=list(self.STAGE_NAMES),
            epoch_seconds=30,
            input_sampling_rate=float(raw.info['sfreq']),
            backend_sampling_rate=float(staging.sf),
            input_units='V',
            channels=names,
            demographics=dict(self.metadata) if self.metadata else None,
            first_sample=int(raw.first_samp),
            first_sample_seconds=float(raw.first_time),
            incomplete_tail_seconds=max(
                0.0, raw.n_times / raw.info['sfreq'] - count * 30
            ),
            protocol='full-record offline pretrained inference; no supervised fitting; centered feature context uses future samples',
        )
        return dict(
            labels=labels,
            probabilities=proba,
            onsets=onsets,
            recording_onsets=onsets + raw.first_time,
            stage_names=self.STAGE_NAMES,
            metadata=provenance,
        )
