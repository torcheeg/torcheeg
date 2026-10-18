from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from test.datasets.test_adfsu import write_condition
from torcheeg.models import EEGNet


PATH = Path(__file__).resolve().parents[2] / 'examples/examples_adfsu_eegnet.py'
SPEC = importlib.util.spec_from_file_location('adfsu_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)
ENTRIES = [('Healthy','Paciente2'),('Healthy','Paciente10'),('AD','Paciente2'),('AD','Paciente10')]


class TestADFSUEEGNetExample(unittest.TestCase):
    def write_source(self, root, entries=ENTRIES, constant=False):
        source={}
        for group,subject in entries:
            for state in ['Eyes_closed','Eyes_open']:
                folder,signal=write_condition(root,group,subject,state)
                if state=='Eyes_open':
                    signal=-2*signal
                if constant:
                    signal[3]=.113596
                for channel,values in zip(EXAMPLE.CHANNELS,signal):
                    np.savetxt(folder/(channel+'.txt'),values,fmt='%.18e')
                source[group+'_'+subject+'_'+state]=(group,signal)
        return source

    def run_example(self, root, state='Eyes_closed'):
        output=root/(state+'.pt')
        args=['example','--root-path',str(root),'--state',state,'--cache-path',str(root/'cache'),
              '--output-path',str(output),'--epochs','1','--batch-size','4']
        with patch('sys.argv',args),redirect_stdout(io.StringIO()):
            EXAMPLE.main()
        return json.loads(output.with_suffix('.pt.json').read_text()),torch.load(output,weights_only=True)

    def test_actual_text_training_natural_folder_holdout_metrics_and_safe_export(self):
        initial={}
        def observe(**kwargs):
            model=EEGNet(**kwargs)
            if not initial:
                initial.update({name:value.clone() for name,value in model.state_dict().items()})
            return model
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=self.write_source(root)
            with patch.object(EXAMPLE,'EEGNet',side_effect=observe) as constructor:
                report,payload=self.run_example(root)
            parts=report['partitions']
            self.assertEqual(parts['train_subject_groups'],['AD_Paciente2','Healthy_Paciente2'])
            self.assertEqual(parts['test_subject_groups'],['AD_Paciente10','Healthy_Paciente10'])
            self.assertEqual(parts['test_groups_by_label'],{'0':'Healthy_Paciente10','1':'AD_Paciente10'})
            self.assertFalse(parts['cross_condition_clinical_linkage_verified'])
            self.assertEqual(len(parts['group_labels']),4)
            self.assertEqual(len(parts['test_records']),2)
            self.assertTrue(all(value.endswith('_Eyes_closed') for value in parts['train_records']+parts['test_records']))
            self.assertFalse(set(parts['train_records']) & set(parts['test_records']))
            self.assertEqual((report['train_windows'],report['test_windows'],report['optimizer_steps']),(16,16,4))
            self.assertEqual((report['train_condition_records'],report['test_condition_records']),(2,2))
            self.assertEqual(report['test_condition_windows'],{'Eyes_closed':16})
            self.assertEqual(json.loads(payload['partitions_json']),parts)
            self.assertEqual(constructor.call_args.kwargs,EXAMPLE.MODEL_CONFIG)
            self.assertEqual(set(json.loads(payload['model_config_json'])),set(inspect.signature(EEGNet).parameters))
            self.assertEqual(len(payload['state_dict']),20)
            self.assertTrue(report['strict_reload_equal'])
            for name in ['block1.0.weight','lin.weight']:
                self.assertFalse(torch.equal(initial[name],payload['state_dict'][name]),name)
            self.assertEqual(report['preprocessing']['states'],['Eyes_closed'])
            self.assertEqual(report['preprocessing']['signal_unit'],'source')
            self.assertEqual(report['preprocessing']['sfreq'],128)
            self.assertIsNone(report['preprocessing']['l_freq'])
            values,targets=[],[]
            for identity in parts['test_records']:
                group,signal=source[identity]
                for start in range(0,1024,128):
                    values.append(signal[:,start:start+128]);targets.append({'Healthy':0,'AD':1}[group])
            values=np.stack(values)
            values=(values-values.mean(-1,keepdims=True))/values.std(-1,keepdims=True)
            model=EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
            model.load_state_dict(payload['state_dict'],strict=True)
            with torch.no_grad():logits=model(torch.tensor(values[:,None],dtype=torch.float32))
            truth=torch.tensor(targets)
            self.assertAlmostEqual(report['test_metrics']['test_loss'],F.cross_entropy(logits,truth).item(),places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'],(logits.argmax(1)==truth).float().mean().item(),places=6)

    def test_selected_condition_has_independent_cache_and_consistent_diagnosis_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_source(root)
            closed,_=self.run_example(root,'Eyes_closed')
            opened,payload=self.run_example(root,'Eyes_open')
            self.assertEqual(opened['partitions']['group_labels'],closed['partitions']['group_labels'])
            self.assertEqual(opened['test_condition_windows'],{'Eyes_open':16})
            self.assertEqual(opened['preprocessing']['states'],['Eyes_open'])
            self.assertTrue(all(value.endswith('_Eyes_open') for value in opened['partitions']['train_records']+opened['partitions']['test_records']))
            self.assertTrue((root/'cache'/'Eyes_open').is_dir())
            self.assertTrue((root/'cache'/'Eyes_closed').is_dir())
            self.assertEqual(json.loads(payload['preprocessing_json']),opened['preprocessing'])

    def test_decimal_constant_channel_is_finite_and_exactly_zero(self):
        seen=[]
        def observe(**kwargs):
            model=EEGNet(**kwargs)
            def check(module,arguments):
                values=arguments[0]
                self.assertTrue(torch.isfinite(values).all())
                self.assertTrue(torch.equal(values[:,:,3],torch.zeros_like(values[:,:,3])))
                seen.append(len(values))
            model.register_forward_pre_hook(check)
            return model
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_source(root,constant=True)
            with patch.object(EXAMPLE,'EEGNet',side_effect=observe):
                report,_=self.run_example(root)
            self.assertTrue(seen)
            self.assertEqual(report['preprocessing']['constant_channels'],'equal_extrema_centered_exactly')

    def test_missing_diagnosis_class_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(EXAMPLE,'ClassifierTrainer') as trainer:
            root=Path(directory);self.write_source(root,entries=ENTRIES[:2])
            with self.assertRaisesRegex(ValueError,'both diagnosis classes'):self.run_example(root)
            trainer.assert_not_called()

    def test_single_folder_in_class_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(EXAMPLE,'ClassifierTrainer') as trainer:
            root=Path(directory);self.write_source(root,entries=ENTRIES[:3])
            with self.assertRaisesRegex(ValueError,'two source folders per diagnosis'):self.run_example(root)
            trainer.assert_not_called()

    def test_existing_model_or_report_is_preserved(self):
        for name in ['Eyes_closed.pt','Eyes_closed.pt.json']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/name).write_bytes(b'existing owned output')
                with self.assertRaisesRegex(ValueError,'fresh output'):self.run_example(root)
                self.assertEqual((root/name).read_bytes(),b'existing owned output')

    def test_invalid_cli_fails_without_reading_source(self):
        for argument,value in [('--epochs','0'),('--batch-size','1'),('--state','both')]:
            with self.subTest(argument=argument,value=value),patch('sys.argv',['example','--root-path','unused',argument,value]),patch.object(EXAMPLE,'ADFSUDataset') as dataset:
                with self.assertRaises(SystemExit):EXAMPLE.main()
                dataset.assert_not_called()


if __name__=='__main__':
    unittest.main()
