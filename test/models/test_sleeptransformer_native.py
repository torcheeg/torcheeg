"""Full TensorFlow 1 oracle generated from unchanged pinned author NN source.

Generate with references/sleeptransformer_fixture.py in a separate Python 3.7
TensorFlow 1.15.5 environment. No TensorFlow dependency enters TorchEEG itself.
"""

import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from torcheeg.models import SleepTransformer


@unittest.skipUnless(
    os.environ.get('TORCHEEG_SLEEPTRANSFORMER_FIXTURE'),
    'Complete pinned TensorFlow 1 oracle supplied separately.',
)
class TestSleepTransformerNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = np.load(
            Path(os.environ['TORCHEEG_SLEEPTRANSFORMER_FIXTURE']), allow_pickle=False
        )
        cls.metadata = json.loads(str(cls.fixture['metadata']))
        if cls.metadata['revision'] != SleepTransformer.SOURCE_REVISION:
            raise AssertionError('Wrong complete native source revision.')
        if (
            cls.metadata['tensorflow'] != '1.15.5'
            or cls.metadata['neural_source_modified']
        ):
            raise AssertionError(
                'Unmodified complete TensorFlow 1.15.5 graph required.'
            )
        if (
            cls.metadata['parameter_count'] != 2448517
            or len(cls.metadata['variable_names']) != 65
        ):
            raise AssertionError('All default NN variables required.')

    @classmethod
    def tearDownClass(cls):
        cls.fixture.close()

    def model(self):
        model = SleepTransformer(dropout=0, attention_dropout=0)
        native = {
            name: self.fixture['state_' + str(i)]
            for i, name in enumerate(self.metadata['variable_names'])
        }
        model.load_native_state_dict(native)
        self.assertEqual(set(native), set(model.native_state_dict()))
        for name, value in model.native_state_dict().items():
            np.testing.assert_array_equal(value.detach().numpy(), native[name])
        return model

    def test_full_native_outputs_context_and_loss(self):
        for training in [False, True]:
            with self.subTest(training=training):
                model = self.model().train(training)
                with torch.no_grad():
                    logits, attention = model(
                        torch.from_numpy(self.fixture['input']), return_attention=True
                    )
                    loss = model.compute_loss(
                        logits, torch.from_numpy(self.fixture['targets'])
                    )
                    ce = model.compute_loss(
                        logits, torch.from_numpy(self.fixture['targets']), 0
                    )
                torch.testing.assert_close(
                    logits,
                    torch.from_numpy(self.fixture['logits']),
                    rtol=3e-5,
                    atol=3e-6,
                )
                torch.testing.assert_close(
                    attention['frame_context'].reshape(40, 29),
                    torch.from_numpy(self.fixture['frame_context']),
                    rtol=3e-5,
                    atol=3e-6,
                )
                torch.testing.assert_close(
                    loss, torch.from_numpy(self.fixture['loss']), rtol=3e-6, atol=3e-6
                )
                torch.testing.assert_close(
                    ce, torch.from_numpy(self.fixture['ce']), rtol=3e-6, atol=3e-6
                )

    def test_native_input_and_every_parameter_gradient(self):
        model = self.model().train()
        x = torch.from_numpy(self.fixture['input']).requires_grad_()
        model.compute_loss(
            model(x), torch.from_numpy(self.fixture['targets'])
        ).backward()
        torch.testing.assert_close(
            x.grad,
            torch.from_numpy(self.fixture['input_gradient']),
            rtol=3e-4,
            atol=3e-6,
        )
        gradients = dict(model.named_parameters())
        mapping = model._native_mapping()
        for i, name in enumerate(self.metadata['variable_names']):
            key, transpose = mapping[name]
            gradient = gradients[key].grad
            self.assertIsNotNone(gradient, name)
            expected = torch.from_numpy(self.fixture['gradient_' + str(i)])
            if transpose:
                expected = expected.T
            torch.testing.assert_close(
                gradient, expected, rtol=3e-4, atol=3e-6, msg=name
            )

    def test_native_full_sgd_update_and_post_update_outputs(self):
        model = self.model()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.metadata['learning_rate']
        )
        x = torch.from_numpy(self.fixture['input'])
        model.compute_loss(
            model(x), torch.from_numpy(self.fixture['targets'])
        ).backward()
        optimizer.step()
        state = model.native_state_dict()
        for i, name in enumerate(self.metadata['variable_names']):
            torch.testing.assert_close(
                state[name],
                torch.from_numpy(self.fixture['after_' + str(i)]),
                rtol=3e-5,
                atol=3e-7,
                msg=name,
            )
        with torch.no_grad():
            torch.testing.assert_close(
                model(x),
                torch.from_numpy(self.fixture['logits_after']),
                rtol=3e-5,
                atol=3e-6,
            )


if __name__ == '__main__':
    unittest.main()
