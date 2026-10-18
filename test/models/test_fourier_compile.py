import os
import unittest

import torch

from torcheeg.models import FourierResample


class TestFourierCompile(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('GITHUB_ACTIONS')=='true' or os.environ.get('TORCHEEG_TEST_COMPILE')=='1', 'Optional compiler regression')
    def test_multiple_constructor_lengths_remain_guarded_constants(self):
        for target,shape in [(64,(2,3,48)),(16,(2,3,48))]:
            with self.subTest(target=target,shape=shape):
                model = FourierResample(target)
                # PyTorch 2.4 has no module-class static marker, so its
                # documented fixed-geometry compile path is explicit.
                older = tuple(int(part) for part in torch.__version__.split('.')[:2]) < (2,6)
                options = {'dynamic':False} if older else {}
                compiled = torch.compile(model,backend='inductor',fullgraph=True,**options)
                x = torch.randn(*shape,requires_grad=True)
                y = x.detach().clone().requires_grad_()
                # FFT input dimensions are explicit constants. Output length
                # must also remain a guarded module constant across instances.
                torch._dynamo.mark_static(y)
                expected = model(x)
                actual = compiled(y)
                coefficient = torch.linspace(-.7,1.1,expected.numel()).reshape_as(expected)
                (expected*coefficient).sum().backward()
                (actual*coefficient).sum().backward()
                self.assertEqual(actual.shape,shape[:-1]+(target,))
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(y.grad,x.grad,rtol=1e-5,atol=1e-6)

    @unittest.skipUnless(os.environ.get('GITHUB_ACTIONS')=='true' or os.environ.get('TORCHEEG_TEST_COMPILE')=='1', 'Optional compiler regression')
    def test_fixed_geometry_multiaxis_windows_keep_fullgraph_input_gradients(self):
        model = FourierResample(16)
        compiled = torch.compile(model, backend='inductor', fullgraph=True, dynamic=False)
        x = torch.randn(2,2,4,12,requires_grad=True)
        y = x.detach().clone().requires_grad_()
        expected = model(x);actual = compiled(y)
        coefficient = torch.linspace(-.7,1.1,expected.numel()).reshape_as(expected)
        (expected*coefficient).sum().backward();(actual*coefficient).sum().backward()
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(y.grad,x.grad,rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
