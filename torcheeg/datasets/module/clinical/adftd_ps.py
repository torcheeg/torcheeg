from ._adftd import _ADFTDDataset


class ADFTDPSDataset(_ADFTDDataset):
    r'''
    A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG (the photostimulation paradigm, ADFTD-PS). The complementary photostimulation recordings use the same numbered cohort as the resting-state ds004504 release, according to the provider. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Ntetska et al.
    - Year: 2025
    - Download URL: https://openneuro.org/datasets/ds006036/versions/1.0.5
    - Reference: Ntetska A, Miltiadous A, Tsipouras M G, et al. A Complementary Dataset of Scalp EEG Recordings Featuring Participants with Alzheimer’s Disease, Frontotemporal Dementia, and Healthy Controls, Obtained from Photostimulation EEG. Data, 2025, 10(5): 64. DOI: 10.3390/data10050064.
    - Paradigm: Whole photostimulation recordings, with diagnosis labels rather than stimulation-frequency targets.
    - Signals: 19-channel scalp EEG (10-20 system) recorded with a Nihon Kohden EEG 2100 device at 500 Hz from 88 subjects (36 Alzheimer's disease, 23 frontotemporal dementia, 29 healthy controls); source SET reference field A1 A2 (the acquisition README also describes a Cz referential montage).
    - Classes: HC (0), AD (1), FTD (2)

    In order to use this dataset, the download folder :obj:`ADFTD-PS` is required (the OpenNeuro BIDS dataset ds006036). The folder should contain the following files:

    .. code-block:: text

        ADFTD-PS/
        ├── participants.tsv
        ├── participants.json
        ├── dataset_description.json
        ├── sub-001
        │   └── eeg
        │       ├── sub-001_task-photomark_channels.tsv
        │       ├── sub-001_task-photomark_eeg.json
        │       └── sub-001_task-photomark_eeg.set
        ├── sub-002
        │   └── eeg
        │       └── ...
        ├── ...
        └── sub-088

    Output values are in volts, converted from the EEGLAB microvolt format.
    PS channel metadata use n/a for units; conversion uses the EEGLAB format
    convention and does not establish independent hardware calibration.
    Photomark events are not used as diagnosis labels or trial boundaries.
    The default pipeline adds notch 50 Hz, FIR 0.5--45 Hz and resamples to
    200 Hz. Set the filter options to None and sfreq=500 to keep native
    released samples. See :doc:`/user_guide/adftd_ps` for source provenance,
    the preprocessed release and discontinuity/time conventions.

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import ADFTDPSDataset
        from torcheeg import transforms

        dataset = ADFTDPSDataset(root_path='./ADFTD-PS',
                                 chunk_size=400,
                                 overlap=0,
                                 num_channel=19,
                                 online_transform=transforms.ToTensor(),
                                 label_transform=transforms.Select('label'))
        print(dataset[0])
        # EEG signal (torch.Tensor[19, 400]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import ADFTDPSDataset
        from torcheeg.datasets.constants import ADFTD_PS_ADJACENCY_MATRIX
        from torcheeg import transforms

        dataset = ADFTDPSDataset(root_path='./ADFTD-PS',
                                 chunk_size=400,
                                 online_transform=transforms.Compose([
                                     transforms.ToG(ADFTD_PS_ADJACENCY_MATRIX)
                                 ]),
                                 label_transform=transforms.Select('label'))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # label (int)

    An example dataset with custom preprocessing parameters:

    .. code-block:: python

        from torcheeg.datasets import ADFTDPSDataset
        from torcheeg import transforms

        dataset = ADFTDPSDataset(root_path='./ADFTD-PS',
                                 chunk_size=400,
                                 overlap=200,
                                 num_channel=19,
                                 sfreq=200,
                                 notch_freqs=50.0,
                                 l_freq=0.5,
                                 h_freq=45.0,
                                 ref_channels='average',
                                 online_transform=transforms.ToTensor(),
                                 label_transform=transforms.Select('label'))
        print(dataset[0])
        # EEG signal (torch.Tensor[19, 400]),
        # label (int)

    Args:
        root_path (str): Path to the unzipped OpenNeuro BIDS dataset folder. (default: :obj:`'./ADFTD-PS'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a recording is used as a single chunk. (default: :obj:`400`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. (default: :obj:`19`)
        sfreq (int): Target sampling rate in Hz for the output EEG signals. The raw recordings are sampled at 500 Hz. (default: :obj:`200`)
        notch_freqs (float, optional): Frequency to remove with a notch filter (e.g., 50 Hz for line noise). If None, the notch filter is not applied. (default: :obj:`50.0`)
        l_freq (float, optional): Low cutoff frequency for the bandpass filter (high-pass). If None, no high-pass filter is applied. (default: :obj:`0.5`)
        h_freq (float, optional): High cutoff frequency for the bandpass filter (low-pass). If None, no low-pass filter is applied. (default: :obj:`45.0`)
        ref_channels (str, optional): Reference channels for re-referencing. Can be 'average' for average reference, or None for no re-referencing. (default: :obj:`None`)
        online_transform (Callable, optional): The transformation of the EEG signals. The input is a :obj:`np.ndarray`, and the output is used as the first value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the output is used as the second value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        after_subject (Callable, optional): The hook performed on the subject to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        preprocessed (bool): Select the provider's derivatives records. Configured filters and resampling still apply; use a separate cache. (default: :obj:`False`)
    '''

    task = 'photomark'
    dataset_name = 'ADFTD-PS'
    derivative_directory = 'derivatives/eeglab'

    def __init__(self, root_path='./ADFTD-PS', chunk_size=400, overlap=0,
                 num_channel=19, sfreq=200, notch_freqs=50., l_freq=.5, h_freq=45.,
                 ref_channels=None, online_transform=None, offline_transform=None,
                 label_transform=None, before_trial=None, after_trial=None,
                 after_subject=None, io_path=None, io_size=10485760, io_mode='lmdb',
                 num_worker=0, verbose=True, preprocessed=False):
        super().__init__(root_path=root_path, chunk_size=chunk_size, overlap=overlap,
                         num_channel=num_channel, sfreq=sfreq, notch_freqs=notch_freqs,
                         l_freq=l_freq, h_freq=h_freq, ref_channels=ref_channels,
                         online_transform=online_transform, offline_transform=offline_transform,
                         label_transform=label_transform, before_trial=before_trial,
                         after_trial=after_trial, after_subject=after_subject,
                         io_path=io_path, io_size=io_size, io_mode=io_mode,
                         num_worker=num_worker, verbose=verbose, preprocessed=preprocessed)

    @classmethod
    def set_records(cls, root_path='./ADFTD-PS', **kwargs):
        return super().set_records(root_path=root_path, **kwargs)
