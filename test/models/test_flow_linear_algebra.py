import io
import math
import warnings
from unittest.mock import patch
import unittest

import torch
import torch.nn.functional as F

from torcheeg.models import BCGlow, BGlow
from torcheeg.models.flow.bglow import InvertibleConv1x1 as UnconditionalConv
from torcheeg.models.flow.bcglow import InvertibleConv1x1 as ConditionalConv


class TestFlowLinearAlgebra(unittest.TestCase):
    def test_orthogonal_initialization_and_reconstruction(self):
        for cls in [UnconditionalConv, ConditionalConv]:
            for lu in [False, True]:
                with self.subTest(cls=cls.__module__, lu=lu):
                    torch.manual_seed(42)
                    layer = cls(4, lu).double()
                    x = torch.randn(2, 4, 3, 5, dtype=torch.double)
                    weight, _ = layer.get_weight(x, False)
                    matrix = weight[:, :, 0, 0]
                    torch.testing.assert_close(matrix.T @ matrix,
                                               torch.eye(4, dtype=torch.double),
                                               rtol=1e-6, atol=1e-6)
                    z, logdet = layer(x, torch.zeros(2, dtype=torch.double))
                    restored, reverse_logdet = layer(z, logdet, reverse=True)
                    torch.testing.assert_close(restored, x)
                    torch.testing.assert_close(reverse_logdet,
                                               torch.zeros_like(logdet))

    def test_known_diagonal_determinant_and_gradients(self):
        diagonal = torch.tensor([2., 3., 0.5, 1.], dtype=torch.double)
        for cls in [UnconditionalConv, ConditionalConv]:
            for lu in [False, True]:
                with self.subTest(cls=cls.__module__, lu=lu):
                    layer = cls(4, lu).double()
                    with torch.no_grad():
                        if lu:
                            layer.p.copy_(torch.eye(4, dtype=torch.double))
                            layer.lower.zero_()
                            layer.upper.zero_()
                            layer.sign_s.fill_(1)
                            layer.log_s.copy_(diagonal.log())
                        else:
                            layer.weight.copy_(diagonal.diag())
                    x = torch.randn(2, 4, 3, 5, dtype=torch.double,
                                    requires_grad=True)
                    z, logdet = layer(x, torch.zeros(2, dtype=torch.double))
                    torch.testing.assert_close(z, x * diagonal.view(1, 4, 1, 1))
                    expected = torch.full((2,), 15 * math.log(3.0),
                                          dtype=torch.double)
                    torch.testing.assert_close(logdet, expected)
                    (z.square().mean() + logdet.mean()).backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    for parameter in layer.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_layer_checkpoint_round_trip(self):
        for cls in [UnconditionalConv, ConditionalConv]:
            for lu in [False, True]:
                with self.subTest(cls=cls.__module__, lu=lu):
                    layer = cls(4, lu)
                    buffer = io.BytesIO()
                    torch.save(layer.state_dict(), buffer)
                    buffer.seek(0)
                    restored = cls(4, lu)
                    restored.load_state_dict(torch.load(buffer, weights_only=True),
                                             strict=True)
                    x = torch.randn(2, 4, 3, 5)
                    expected, expected_logdet = layer(x, torch.zeros(2))
                    actual, actual_logdet = restored(x, torch.zeros(2))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual_logdet, expected_logdet,
                                               rtol=0, atol=0)

    def test_legacy_lu_fallback_when_modern_entry_point_is_absent(self):
        for cls in [UnconditionalConv, ConditionalConv]:
            torch.manual_seed(42)
            expected = cls(4, True).state_dict()
            with patch.object(torch.linalg, 'lu'):
                delattr(torch.linalg, 'lu')
                torch.manual_seed(42)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='torch.lu is deprecated')
                    try:
                        actual = cls(4, True).state_dict()
                    except RuntimeError as error:
                        if 'removed' in str(error):
                            self.skipTest('Legacy LU is removed in this runtime.')
                        raise
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name],
                                           rtol=0, atol=0)

    def test_flow_optimization_and_sampling(self):
        for cls in [BGlow, BCGlow]:
            for lu in [False, True]:
                with self.subTest(cls=cls.__name__, lu=lu):
                    model = cls(in_channels=2, grid_size=(8, 8), hidden_channels=8,
                                num_steps=2, num_blocks=1, LU_decomposed=lu)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    x = torch.randn(2, 2, 8, 8)
                    y = torch.tensor([0, 1])
                    if cls is BCGlow:
                        _, nll, logits = model(x, y)
                        loss = nll.mean() + F.cross_entropy(logits, y)
                    else:
                        _, nll = model(x)
                        loss = nll.mean()
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                    optimizer.step()
                    with torch.no_grad():
                        sample = model.sample(y if cls is BCGlow else 2)
                    self.assertEqual(tuple(sample.shape), (2, 2, 8, 8))
                    self.assertTrue(torch.isfinite(sample).all())


if __name__ == '__main__':
    unittest.main()
