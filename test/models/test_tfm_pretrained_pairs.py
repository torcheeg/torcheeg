from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from torcheeg.models import TFMClassifier, TFMEncoder, TFMTokenizer


class TestTFMPretrainedPairs(unittest.TestCase):
    def test_complete_mtp_head_mask_row_and_vocabulary_gradients(self):
        model = TFMEncoder(num_classes=8192).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mtp.pt'
            torch.save(model.state_dict(), path)
            restored = TFMEncoder.from_pretrained(path, num_classes=8192).eval()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, restored.state_dict()[name], rtol=0, atol=0
                )
            ids = torch.tensor([[[8192, 12, 713], [91, 8192, 6500]]])
            logits = restored.masked_prediction(ids)
            torch.testing.assert_close(
                logits, model.masked_prediction(ids), rtol=0, atol=0
            )
            self.assertEqual(logits.shape, (1, 6, 8192))
            targets = torch.tensor([12, 1, 713, 91, 213, 6500])
            nn.functional.cross_entropy(logits.reshape(-1, 8192), targets).backward()
            self.assertGreater(
                restored.eeg_token_embedding.weight.grad[8192].abs().sum().item(), 0.0
            )
            self.assertTrue(
                (restored.classification_head.weight.grad.abs().sum(-1) > 0).all()
            )
            self.assertTrue(
                all(
                    torch.isfinite(p.grad).all()
                    for p in restored.parameters()
                    if p.grad is not None
                )
            )
            with self.assertRaisesRegex(ValueError, 'shape/type/value'):
                TFMEncoder.from_pretrained(path)

    def test_explicit_complete_pair_prefix_and_frozen_teacher(self):
        tokenizer, encoder = (
            TFMTokenizer(reconstruction_head='mlp'),
            TFMEncoder(num_classes=8192),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tpath, epath = root / 'tokenizer.pt', root / 'encoder.pt'
            torch.save(
                {'module.' + n: v for n, v in tokenizer.state_dict().items()}, tpath
            )
            torch.save(
                {'module.' + n: v for n, v in encoder.state_dict().items()}, epath
            )
            model = TFMClassifier.from_pretrained(
                tpath,
                epath,
                tokenizer_prefix='module.',
                encoder_prefix='module.',
                freeze_encoder=True,
            )
            self.assertEqual(len(model.tokenizer.state_dict()), 191)
            self.assertEqual(len(model.encoder.state_dict()), 59)
            model.train()
            self.assertFalse(model.tokenizer.training)
            self.assertFalse(model.encoder.training)
            logits = model(torch.randn(2, 2, 800))
            nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
            self.assertIsNotNone(model.classifier.weight.grad)
            self.assertTrue(all(p.grad is None for p in model.tokenizer.parameters()))
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            for prefix, component in [('tokenizer', tokenizer), ('encoder', encoder)]:
                for name, value in component.state_dict().items():
                    torch.testing.assert_close(
                        value, model.state_dict()[prefix + '.' + name], rtol=0, atol=0
                    )
            with self.assertRaisesRegex(ValueError, 'Incomplete TFM'):
                TFMClassifier.from_pretrained(
                    tpath,
                    epath,
                    variant='single-chbmit',
                    tokenizer_prefix='module.',
                    encoder_prefix='module.',
                )
            for invalid in ('unknown', None, []):
                with self.assertRaisesRegex(ValueError, 'variant'):
                    TFMClassifier.from_pretrained(tpath, epath, variant=invalid)


if __name__ == '__main__':
    unittest.main()
