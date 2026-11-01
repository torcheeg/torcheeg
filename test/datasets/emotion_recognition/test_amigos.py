import unittest

from torcheeg.datasets import AMIGOSDataset


class TestAmigos(unittest.TestCase):

    def test_amigos_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(AMIGOSDataset.fake_record(**kwargs))
        item = next(AMIGOSDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (14, 128))


if __name__ == '__main__':
    unittest.main()
