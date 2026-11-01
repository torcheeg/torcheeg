import importlib
import unittest


class TestFlowExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.models.flow')
        expected = {'BCGlow': 'torcheeg.models.flow.bcglow',
                    'BGlow': 'torcheeg.models.flow.bglow'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
