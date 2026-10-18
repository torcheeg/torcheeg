import copy
import os
import unittest

import torch

from torcheeg.models import (CBraMod, CBraModClassifier, add_cbramod_adapters,
                             add_cbramod_prompts, add_cbramod_lora)


@unittest.skipUnless(os.environ.get('TORCHEEG_TEST_COMPILE') == '1' or
                     os.environ.get('GITHUB_ACTIONS') == 'true',
                     'Enable TORCHEEG_TEST_COMPILE=1 with a working C++ compiler; GitHub Actions runs this check.')
class TestCBraModLoRACompile(unittest.TestCase):
    def test_fullgraph_training_outputs_inputs_and_parameter_gradients(self):
        torch.manual_seed(42)
        for variant in ('feed_forward', 'query_only', 'all_projections', 'combined'):
            with self.subTest(variant=variant):
                backbone = CBraMod(patch_size=25, hid_channels=16, mlp_channels=32,
                    depth=1, num_heads=4, dropout=0., position_kernel=(3, 3))
                model = CBraModClassifier(backbone, 3, 128, num_patches=4, dropout=0.)
                if variant == 'feed_forward':
                    targets = ('linear1', 'linear2')
                elif variant == 'query_only':
                    targets = ('self_attn_s.q_proj',)
                else:
                    targets = ('linear1', 'linear2') + tuple(axis + '.' + name
                        for axis in ('self_attn_s', 'self_attn_t')
                        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'))
                add_cbramod_lora(model, rank=2, alpha=5., dropout=0., target_modules=targets)
                if variant == 'combined':
                    add_cbramod_prompts(model, num_prompts=2)
                    add_cbramod_adapters(model, adapter_size=4, dropout=0.)
                with torch.no_grad():
                    model.task_head.output.weight.mul_(10.)
                    for name, parameter in model.named_parameters():
                        if name.endswith('lora_B') or name.endswith('up_project.weight'):
                            parameter.normal_(std=.02)
                reference = copy.deepcopy(model)
                x = torch.randn(2, 3, 128, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                mask = torch.zeros(2, 3, 4, dtype=torch.bool)
                mask[:, :, 0] = True
                expected = reference(x, mask=mask)
                expected.square().mean().backward()
                actual = torch.compile(model, backend='inductor', fullgraph=True)(y, mask=mask)
                actual.square().mean().backward()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(y.grad, x.grad, rtol=1e-3, atol=1e-5)
                self.assertTrue(torch.isfinite(y.grad).all())
                torch.testing.assert_close(y.grad[:, :, :32], torch.zeros_like(y.grad[:, :, :32]), rtol=0, atol=0)
                for (name, left), (_, right) in zip(model.named_parameters(), reference.named_parameters()):
                    torch.testing.assert_close(left.grad, right.grad, rtol=1e-3, atol=1e-5, msg=name)


if __name__ == '__main__':
    unittest.main()
