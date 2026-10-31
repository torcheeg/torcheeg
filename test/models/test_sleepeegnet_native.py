"""Full author TensorFlow 1 default and bidirectional CNN/seq2seq oracles."""

import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch
from torcheeg.models import SleepEEGNet


@unittest.skipUnless(
    os.environ.get('TORCHEEG_SLEEPEEGNET_FIXTURES'),
    'Complete pinned native TensorFlow 1 oracles supplied separately.',
)
class TestSleepEEGNetNative(unittest.TestCase):
    def cases(self):
        root = Path(os.environ['TORCHEEG_SLEEPEEGNET_FIXTURES'])
        for bidirectional, name in [
            (False, 'unidirectional.npz'),
            (True, 'bidirectional.npz'),
        ]:
            with np.load(root / name, allow_pickle=False) as fixture:
                meta = json.loads(str(fixture['metadata']))
                self.assertEqual(meta['revision'], SleepEEGNet.SOURCE_REVISION)
                self.assertEqual(
                    meta['source_sha256'],
                    '496bdcb027b3dbc744f249d53f82388a833617204c881223074fd16e8b07f6bb',
                )
                self.assertFalse(meta['neural_functions_modified'])
                self.assertEqual(meta['tensorflow'], '1.15.5')
                self.assertEqual(meta['bidirectional'], bidirectional)
                model = SleepEEGNet(bidirectional=bidirectional, dropout=0)
                names = meta['variable_names']
                self.assertEqual(set(model.native_state_dict()), set(names))
                model.load_native_state_dict(
                    {name: fixture['state_' + str(i)] for i, name in enumerate(names)}
                )
                for i, key in enumerate(names):
                    np.testing.assert_array_equal(
                        model.native_state_dict()[key].detach().numpy(),
                        fixture['state_' + str(i)],
                    )
                yield bidirectional, model, fixture, meta

    def test_complete_native_teacher_forcing_attention_class_balanced_loss_and_greedy(
        self,
    ):
        for bidirectional, model, fixture, _ in self.cases():
            with self.subTest(bidirectional=bidirectional):
                x = torch.from_numpy(fixture['input']).unsqueeze(2)
                for training in [False, True]:
                    model.train(training)
                    with torch.no_grad():
                        logits, attention = model(
                            x,
                            torch.from_numpy(fixture['decoder_inputs']),
                            return_attention=True,
                        )
                        loss = model.compute_loss(
                            logits, torch.from_numpy(fixture['targets'])
                        )
                    torch.testing.assert_close(
                        logits,
                        torch.from_numpy(fixture['logits']),
                        rtol=1e-4,
                        atol=2e-7,
                    )
                    torch.testing.assert_close(
                        attention,
                        torch.from_numpy(fixture['attention']),
                        rtol=1e-4,
                        atol=2e-7,
                    )
                    torch.testing.assert_close(
                        loss, torch.from_numpy(fixture['loss']), rtol=3e-6, atol=3e-6
                    )
                model.eval()
                generated = model.generate(x)
                self.assertTrue(
                    torch.equal(
                        generated['tokens'], torch.from_numpy(fixture['tokens']).long()
                    )
                )
                expected = []
                for row in fixture['tokens']:
                    endings = np.flatnonzero(row == model.end_token)
                    expected.append(int(endings[0] + 1) if len(endings) else len(row))
                self.assertEqual(generated['lengths'].tolist(), expected)

    def test_complete_native_input_and_all_parameter_gradients(self):
        for bidirectional, model, fixture, meta in self.cases():
            with self.subTest(bidirectional=bidirectional):
                x = torch.from_numpy(fixture['input']).unsqueeze(2).requires_grad_()
                loss = model.compute_loss(
                    model(x, torch.from_numpy(fixture['decoder_inputs'])),
                    torch.from_numpy(fixture['targets']),
                )
                loss.backward()
                expected_input = torch.from_numpy(fixture['input_gradient']).unsqueeze(
                    2
                )
                torch.testing.assert_close(x.grad, expected_input, rtol=1e-3, atol=2e-8)
                self.assertLess(
                    float((x.grad - expected_input).norm()),
                    float(expected_input.norm()) * 1e-3 + 1e-8,
                )
                parameters = dict(model.named_parameters())
                for i, native in enumerate(meta['variable_names']):
                    key, order = model._native_mapping()[native]
                    expected = torch.from_numpy(fixture['gradient_' + str(i)])
                    if order:
                        expected = expected.permute(order)
                    actual = parameters[key].grad
                    self.assertIsNotNone(actual, native)
                    torch.testing.assert_close(
                        actual, expected, rtol=1e-3, atol=1e-7, msg=native
                    )
                    self.assertLess(
                        float((actual - expected).norm()),
                        float(expected.norm()) * 1e-3 + 1e-7,
                        native,
                    )

    def test_complete_native_sgd_and_post_update_teacher_and_greedy(self):
        for bidirectional, model, fixture, meta in self.cases():
            with self.subTest(bidirectional=bidirectional):
                x = torch.from_numpy(fixture['input']).unsqueeze(2)
                decoder = torch.from_numpy(fixture['decoder_inputs'])
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=meta['learning_rate']
                )
                model.compute_loss(
                    model(x, decoder), torch.from_numpy(fixture['targets'])
                ).backward()
                optimizer.step()
                for i, name in enumerate(meta['variable_names']):
                    expected = torch.from_numpy(
                        fixture['state_' + str(i)] + fixture['delta_' + str(i)]
                    )
                    torch.testing.assert_close(
                        model.native_state_dict()[name],
                        expected,
                        rtol=1e-5,
                        atol=3e-7,
                        msg=name,
                    )
                with torch.no_grad():
                    torch.testing.assert_close(
                        model(x, decoder),
                        torch.from_numpy(fixture['logits_after']),
                        rtol=1e-4,
                        atol=2e-7,
                    )
                model.eval()
                self.assertTrue(
                    torch.equal(
                        model.generate(x)['tokens'],
                        torch.from_numpy(fixture['tokens_after']).long(),
                    )
                )


if __name__ == '__main__':
    unittest.main()
