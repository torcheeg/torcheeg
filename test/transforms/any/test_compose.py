import unittest

from torcheeg.transforms import Compose


def add_one(value):
    return value + 1


def double(value):
    return value * 2


class TestComposeTransforms(unittest.TestCase):
    def test_order_and_flattening(self):
        transform = Compose([add_one, Compose([double, add_one])])
        self.assertEqual(transform(3), 9)
        self.assertEqual(len(transform.transforms), 3)

    def test_dictionary_pipeline(self):
        transform = Compose([
            lambda sample: {**sample, 'eeg': sample['eeg'] + 1},
            lambda sample: {'eeg': sample['eeg'] *
                            2, 'labels': sample['info']['label']},
        ])
        self.assertEqual(transform({'eeg': 2, 'info': {'label': 1}}),
                         {'eeg': 6, 'labels': 1})

    def test_identity_and_validation(self):
        value = object()
        self.assertIs(Compose([])(value), value)
        with self.assertRaises(TypeError):
            Compose([None])
