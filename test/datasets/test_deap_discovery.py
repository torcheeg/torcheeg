import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset


class TestDEAPDiscovery(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name, 'source')
        self.root.mkdir()
        self.dataset = DEAPDataset.__new__(DEAPDataset)

    def test_macos_sidecars_and_unrelated_entries_are_ignored(self):
        for name in ('.DS_Store', '._s01.dat', '.hidden.dat', 'README.txt', 's02.dat', 's01.DAT'):
            (self.root / name).touch()
        (self.root / 'nested.dat').mkdir()
        records = self.dataset.set_records(str(self.root))
        self.assertEqual(records, ['s01.DAT', 's02.dat'])

    def test_discovery_is_sorted_and_accepts_custom_visible_record_names(self):
        for name in ('s03.dat', 'custom_subject.dat', 's01.dat'):
            (self.root / name).touch()
        self.assertEqual(self.dataset.set_records(str(self.root)),
                         ['custom_subject.dat', 's01.dat', 's03.dat'])

    def test_empty_or_sidecar_only_directories_have_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'No DEAP .dat files'):
            self.dataset.set_records(str(self.root))
        (self.root / '.DS_Store').write_bytes(b'\x00\x00')
        with self.assertRaisesRegex(ValueError, 'No DEAP .dat files'):
            self.dataset.set_records(str(self.root))

    def test_mixed_directory_processes_real_pickle_records_and_cached_signals(self):
        samples = np.arange(2 * 40 * 640, dtype=float).reshape(2, 40, 640)
        labels = np.array([[2., 3., 4., 5.], [6., 7., 8., 9.]])
        (self.root / '.DS_Store').write_bytes(b'\x00\x00')
        (self.root / '._s01.dat').write_bytes(b'not a pickle')
        (self.root / 'README.txt').write_text('sidecar')
        (self.root / 'folder.dat').mkdir()
        for subject in ('s02', 's01'):
            with (self.root / (subject + '.dat')).open('wb') as file:
                pickle.dump({'data': samples, 'labels': labels}, file)
        parameters = dict(root_path=str(self.root),
                          io_path=str(Path(self.directory.name, 'cache')),
                          io_mode='pickle', verbose=False,
                          label_transform=transforms.Select('valence'))
        dataset = DEAPDataset(**parameters)
        self.assertEqual(len(dataset), 8)
        self.assertEqual(set(dataset.info['subject_id']), {'s01.dat', 's02.dat'})
        eeg, label = dataset[0]
        np.testing.assert_array_equal(eeg, samples[0, :32, 384:512])
        self.assertEqual(label, 2.)
        self.assertEqual(dataset[2][1], 6.)
        reused = DEAPDataset(**parameters)
        np.testing.assert_array_equal(reused[0][0], eeg)
        self.assertEqual(len(reused), 8)


if __name__ == '__main__':
    unittest.main()
