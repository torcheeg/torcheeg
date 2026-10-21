"""Complete separately supplied native SleepFM diagnosis/loss/update oracle.

Inputs/outcomes here are explicitly synthetic. This is not clinical validation.
"""

import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import unittest

import torch

from torcheeg.models import SleepFMDiagnosis, SleepFMSurvivalObjective


@unittest.skipUnless(
    os.environ.get('TORCHEEG_SLEEPFM_BACKEND_ROOT'),
    'Explicit complete trusted original SleepFM source and official weights required.',
)
class TestSleepFMSurvivalNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ['TORCHEEG_SLEEPFM_BACKEND_ROOT']).resolve()
        revision = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True
        ).strip()
        if revision != '2bcbae04c3592f61352addb7ac3d4193f0a3ca25':
            raise ValueError('Require the pinned complete author release.')
        subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
        source = root / 'sleepfm/pipeline/finetune_diagnosis_coxph.py'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != '810fd6308f3834a995e3c10d16397825d1039aba8ac1f74c026d3690013dadf9':
            raise ValueError('Original native criterion source differs.')
        tree = ast.parse(source.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'cox_ph_loss'
        ]
        if len(functions) != 1:
            raise ValueError('Require the unchanged original loss function.')
        # Evaluate the literal original function body, without importing the
        # unrelated pipeline CLI/logging/data/network dependencies.
        scope = {'torch': torch}
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'),
            scope,
        )
        cls.native_loss = staticmethod(scope['cox_ph_loss'])
        cls.criterion_id = revision + ':' + digest
        spec = importlib.util.spec_from_file_location(
            'sleepfm_survival_complete_native', root / 'sleepfm/models/models.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        folder = root / 'sleepfm/checkpoints/model_diagnosis'
        config = json.loads((folder / 'config.json').read_text())
        cls.model_type = getattr(module, config['model'])
        cls.parameters = config['model_params']
        cls.checkpoint = folder / 'best.pth'
        cls.names = ['synthetic_endpoint_' + str(i) for i in range(1065)]

    def build(self, ties):
        diagnosis = SleepFMDiagnosis.from_pretrained(
            self.checkpoint,
            self.model_type(**self.parameters),
            ['BAS', 'RESP'],
            ['explicit_encoded_column0', 'explicit_encoded_column1'],
        )
        self.assertEqual(len(diagnosis.backend.state_dict()), 35)
        self.assertEqual(
            sum(v.numel() for v in diagnosis.backend.state_dict().values()), 1792937
        )
        model = SleepFMSurvivalObjective(
            diagnosis,
            self.names,
            ties=ties,
            native_criterion=self.native_loss if ties == 'native' else None,
            criterion_id=self.criterion_id if ties == 'native' else None,
        )
        self.assertEqual(len(model.state_dict()), 37)
        return model

    def inputs(self, tied=False):
        torch.manual_seed(17)
        x = torch.randn(4, 2, 5, 128)
        demo = torch.randn(4, 2)
        mask = torch.zeros(4, 2, 5, dtype=torch.bool)
        mask[1, 1] = True
        mask[1, :, 3:] = True
        times = ((torch.arange(4).unsqueeze(1) + torch.arange(1065)) % 4 + 1).float()
        if tied:
            times[:2] = 4
        events = (torch.arange(4).unsqueeze(1) + torch.arange(1065)) % 3 != 0
        return x, demo, mask, times, events

    def test_full_official_native_losses_all_gradients_and_adamw_updates(self):
        for ties in ['breslow', 'native']:
            for training in [False, True]:
                with self.subTest(ties=ties, training=training):
                    model = self.build(ties).train(training)
                    reference = copy.deepcopy(model.model.backend).train(training)
                    x, demo, mask, times, events = self.inputs(tied=ties == 'native')
                    left, right = x.clone().requires_grad_(), x.clone().requires_grad_()
                    a, b = demo.clone().requires_grad_(), demo.clone().requires_grad_()
                    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
                    ref_opt = torch.optim.AdamW(reference.parameters(), lr=0.001)
                    torch.manual_seed(31)
                    actual = model(left, a, times, events, mask)
                    torch.manual_seed(31)
                    expected = reference(right, mask, b)
                    expected_loss = self.native_loss(expected, times, events)
                    torch.testing.assert_close(
                        actual['log_risks'], expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual['loss'], expected_loss, rtol=0, atol=0
                    )
                    actual['loss'].backward()
                    expected_loss.backward()
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                    for (name, p), (other, q) in zip(
                        model.model.backend.named_parameters(),
                        reference.named_parameters(),
                    ):
                        self.assertEqual(name, other)
                        self.assertEqual(p.grad is None, q.grad is None, name)
                        if p.grad is not None:
                            torch.testing.assert_close(
                                p.grad, q.grad, rtol=0, atol=0, msg=name
                            )
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                    opt.step()
                    ref_opt.step()
                    for name, value in reference.state_dict().items():
                        torch.testing.assert_close(
                            model.model.backend.state_dict()[name],
                            value,
                            rtol=0,
                            atol=0,
                            msg=name,
                        )

    def test_complete_official_dropout_optimizer_scheduler_rng_continuation(self):
        model = self.build('breslow').train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.1, patience=10
        )
        x, demo, mask, times, events = self.inputs(tied=True)
        torch.manual_seed(49)
        for _ in range(2):
            opt.zero_grad()
            loss = model(x, demo, times, events, mask)['loss']
            loss.backward()
            opt.step()
            scheduler.step(loss.item())
        stream = io.BytesIO()
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=torch.get_rng_state(),
            ),
            stream,
        )
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = self.build('breslow').train()
        restored.to_empty(device='cpu')
        restored.load_state_dict(saved['model'], strict=True)
        second = torch.optim.AdamW(restored.parameters(), lr=0.001)
        second.load_state_dict(saved['optimizer'])
        second_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            second, factor=0.1, patience=10
        )
        second_scheduler.load_state_dict(saved['scheduler'])
        states = []
        losses = []
        for net, optimizer, schedule in [
            (model, opt, scheduler),
            (restored, second, second_scheduler),
        ]:
            torch.set_rng_state(saved['rng'])
            optimizer.zero_grad()
            loss = net(x, demo, times, events, mask)['loss']
            losses.append(loss.detach())
            loss.backward()
            optimizer.step()
            schedule.step(loss.item())
            states.append(torch.get_rng_state())
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, restored.state_dict()[name], rtol=0, atol=0, msg=name
            )
        for a, b in zip(opt.state.values(), second.state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
        self.assertEqual(scheduler.state_dict(), second_scheduler.state_dict())


if __name__ == '__main__':
    unittest.main()
