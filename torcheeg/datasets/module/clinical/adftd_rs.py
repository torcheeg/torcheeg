from ._adftd import _ADFTDDataset


class ADFTDRSDataset(_ADFTDDataset):
    r'''
    A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG (the resting-state, eyes-closed paradigm, ADFTD-RS). The recordings were acquired from the AHEPA General Hospital of Thessaloniki and are publicly released on OpenNeuro. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Miltiadous et al.
    - Year: 2023
    - Download URL: https://openneuro.org/datasets/ds004504/versions/1.0.8
    - Reference: Miltiadous A, Tzimourta K D, Afrantou T, et al. A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG[J]. Data, 2023, 8(6): 95.
    - Paradigm: Resting-state, eyes-closed routine clinical EEG (RS) - one of the two ADFTD paradigms.
    - Signals: 19-channel scalp EEG (10-20 system) recorded with a Nihon Kohden EEG 2100 device at 500 Hz from 88 subjects (36 Alzheimer's disease, 23 frontotemporal dementia, 29 healthy controls); source SET reference field A1 A2 (the acquisition README also describes a Cz referential montage).
    - Classes: HC (0), AD (1), FTD (2)

    In order to use this dataset, the download folder :obj:`ADFTD-RS` is required (the OpenNeuro BIDS dataset ds004504). The folder should contain the following files:

    .. code-block:: text

        ADFTD-RS/
        ├── participants.tsv
        ├── participants.json
        ├── dataset_description.json
        ├── sub-001
        │   └── eeg
        │       ├── sub-001_task-eyesclosed_channels.tsv
        │       ├── sub-001_task-eyesclosed_eeg.json
        │       └── sub-001_task-eyesclosed_eeg.set
        ├── sub-002
        │   └── eeg
        │       └── ...
        ├── ...
        └── sub-088

    Output values are in volts, converted from the EEGLAB microvolt format.
    The default pipeline adds notch 50 Hz, FIR 0.5--45 Hz and resamples to
    200 Hz. Set the filter options to None and sfreq=500 to keep native
    released samples. See :doc:`/user_guide/adftd_rs` for source provenance,
    the preprocessed release and discontinuity/time conventions.

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import ADFTDRSDataset
        from torcheeg import transforms

        dataset = ADFTDRSDataset(root_path='./ADFTD-RS',
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

        from torcheeg.datasets import ADFTDRSDataset
        from torcheeg.datasets.constants import ADFTD_RS_ADJACENCY_MATRIX
        from torcheeg import transforms

        dataset = ADFTDRSDataset(root_path='./ADFTD-RS',
                                 chunk_size=400,
                                 online_transform=transforms.Compose([
                                     transforms.ToG(ADFTD_RS_ADJACENCY_MATRIX)
                                 ]),
                                 label_transform=transforms.Select('label'))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # label (int)

    An example dataset with custom preprocessing parameters:

    .. code-block:: python

        from torcheeg.datasets import ADFTDRSDataset
        from torcheeg import transforms

        dataset = ADFTDRSDataset(root_path='./ADFTD-RS',
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
        root_path (str): Path to the unzipped OpenNeuro BIDS dataset folder. (default: :obj:`'./ADFTD-RS'`)
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
