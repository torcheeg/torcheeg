import unittest

from torcheeg.transforms import Select, Binary, BinariesToCategory, StringToInt


class TestLabelTransforms(unittest.TestCase):
    def test_select(self):
        info = {'valence': 4.5, 'arousal': 5.5, 'subject_id': 7}
        transformed_label = Select(key='valence')(y=info)
        self.assertEqual(transformed_label['y'], 4.5)

        transformed_label = Select(key=['valence', 'arousal'])(y=info)
        self.assertEqual(transformed_label['y'], [4.5, 5.5])

    def test_binary(self):
        self.assertEqual(Binary(threshold=5.0)(y=4.5)['y'], 0)
        self.assertEqual(Binary(threshold=5.0)(y=5.5)['y'], 1)
        self.assertEqual(Binary(threshold=5.0)(y=[4.5, 5.5])['y'], [0, 1])

    def test_binaries_to_multiple(self):
        self.assertEqual(BinariesToCategory()(y=[0, 0])['y'], 0)
        self.assertEqual(BinariesToCategory()(y=[0, 1])['y'], 1)
        self.assertEqual(BinariesToCategory()(y=[1, 0])['y'], 2)
        self.assertEqual(BinariesToCategory()(y=[1, 1])['y'], 3)

    def test_string_to_int(self):
        transform = StringToInt()

        self.assertEqual(transform(y='None')['y'], 0)
        self.assertEqual(transform(y='sub001')['y'], 1)

        self.assertEqual(transform(y=['None', 'sub001'])['y'], [0, 1])
        self.assertEqual(transform(y=['sub001', '4'])['y'], [1, 4])


if __name__ == '__main__':
    unittest.main()
