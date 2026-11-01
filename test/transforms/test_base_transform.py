import unittest

from torcheeg.transforms.base_transform import (BaseTransform, EEGTransform,
                                                LabelTransform)


class TestBaseTransform(unittest.TestCase):
    def test_required_apply(self):
        for cls in (BaseTransform, EEGTransform, LabelTransform):
            with self.assertRaisesRegex(NotImplementedError, 'Method apply is not implemented in class ' + cls.__name__):
                cls()(1)

    def test_apply_and_repr(self):
        class Add(BaseTransform):
            def apply(self, value):
                return value + 1

            @property
            def repr_body(self):
                return {'amount': 1}
        self.assertEqual(Add()(4), 5)
        self.assertEqual(repr(Add()), 'Add(amount=1)')
