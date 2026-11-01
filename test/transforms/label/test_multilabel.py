import unittest

from torcheeg.transforms import Multilabel, Select


class TestMultilabelTransforms(unittest.TestCase):

    def test_multilabel(self):
        t = Multilabel([Select('valence'), Select('subject_id')])
        self.assertEqual(
            t({'valence': 4.5, 'arousal': 5.5, 'subject_id': 7}), [4.5, 7])


if __name__ == '__main__':
    unittest.main()
