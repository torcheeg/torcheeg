"""Survival cohort identity, named alignment, file isolation and batching tests."""

import copy
import csv
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from torcheeg.datasets import SleepFMSurvivalDataset, collate_sleepfm_survival
from test.datasets._sleepfm_cohort import (
    create_cohort,
    replace_embedding,
    save_manifest,
)


class TestSleepFMSurvivalDataset(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths, self.manifest, self.expected = create_cohort(self.directory.name)

    def build(self, split='train'):
        return SleepFMSurvivalDataset(**self.paths, split=split)

    def rewrite_csv(self, name, change):
        path = self.paths[name]
        with path.open(newline='') as stream:
            rows = list(csv.reader(stream))
        change(rows)
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)

    def test_named_csv_alignment_string_ids_and_all_endpoints(self):
        dataset = self.build()
        self.assertEqual(len(dataset), 6)
        self.assertEqual(
            dataset.info['study_id'].tolist(), [f'{i:04d}' for i in range(6)]
        )
        item = dataset[0]
        self.assertEqual(item['study_id'], '0000')
        self.assertEqual(item['subject_id'], 'patient_0000')
        self.assertEqual(len(dataset.endpoint_names), 1065)
        expected = self.expected['0000']
        torch.testing.assert_close(
            item['event_times'], torch.from_numpy(expected['times']), rtol=0, atol=0
        )
        torch.testing.assert_close(
            item['is_event'], torch.from_numpy(expected['events']), rtol=0, atol=0
        )
        torch.testing.assert_close(
            item['demographics'],
            torch.from_numpy(expected['demographics']),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            item['embeddings'][0],
            torch.from_numpy(expected['arrays']['BAS']),
            rtol=0,
            atol=0,
        )
        self.assertTrue(item['padding_mask'][1].all())
        self.assertEqual(torch.count_nonzero(item['embeddings'][1]), 0)
        dataset.validate_files()
        self.assertEqual(
            self.build('validation').info['study_id'].tolist(), ['0006', '0007', '0008']
        )
        self.assertEqual(self.build('test').info['study_id'].tolist(), ['0009', '0010'])

    def test_collation_keeps_full_lengths_modalities_and_subject_order(self):
        dataset = self.build()
        items = [dataset[i] for i in [0, 2, 1]]
        batch = collate_sleepfm_survival(items)
        self.assertEqual(batch['embeddings'].shape, (3, 4, 4, 128))
        self.assertEqual(batch['study_id'], ['0000', '0002', '0001'])
        self.assertEqual(batch['event_times'].dtype, torch.float64)
        self.assertEqual(batch['is_event'].dtype, torch.bool)
        for row, item in enumerate(items):
            length = item['embeddings'].shape[1]
            torch.testing.assert_close(
                batch['embeddings'][row, :, :length], item['embeddings'], rtol=0, atol=0
            )
            self.assertTrue(batch['padding_mask'][row, :, length:].all())
            self.assertEqual(
                torch.count_nonzero(batch['embeddings'][row, :, length:]), 0
            )
            torch.testing.assert_close(
                batch['padding_mask'][row, :, :length], item['padding_mask']
            )
        with self.assertRaisesRegex(ValueError, 'repeat'):
            collate_sleepfm_survival([dataset[0], dataset[0]])
        changed = copy.deepcopy(items[0])
        changed['schema_fingerprint'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'different.*schemas'):
            collate_sleepfm_survival([items[1], changed])

    def test_patient_study_path_and_content_overlap_are_rejected_globally(self):
        original = copy.deepcopy(self.manifest)
        for field in ['subject_id', 'study_id', 'path', 'sha256']:
            with self.subTest(field=field):
                manifest = copy.deepcopy(original)
                manifest['records'][-1][field] = manifest['records'][0][field]
                save_manifest(self.paths, manifest)
                with self.assertRaises(ValueError):
                    self.build()
        save_manifest(self.paths, original)

    def test_input_tables_reject_missing_or_ambiguous_identity_and_columns(self):
        originals = {
            name: Path(path).read_bytes()
            for name, path in self.paths.items()
            if name != 'manifest_path'
        }
        changes = [
            (
                'event_times_path',
                lambda rows: rows[0].__setitem__(0, 'missing_endpoint'),
            ),
            ('is_event_path', lambda rows: rows.append(rows[-1].copy())),
            (
                'demographics_path',
                lambda rows: rows.__setitem__(slice(None), rows[:-1]),
            ),
            ('event_times_path', lambda rows: rows[0].__setitem__(0, rows[0][1])),
            ('event_times_path', lambda rows: rows[-1].append('extra')),
            ('is_event_path', lambda rows: rows[-1].__setitem__(-1, 'not_selected')),
        ]
        for name, change in changes:
            with self.subTest(name=name, change=change):
                self.rewrite_csv(name, change)
                with self.assertRaises(ValueError):
                    self.build()
                Path(self.paths[name]).write_bytes(originals[name])

    def test_selected_missing_nonfinite_negative_and_nonbinary_values(self):
        for name, values in [
            ('event_times_path', ['', 'nan', 'inf', '-1']),
            ('is_event_path', ['', 'NaN', '2', '-1']),
            ('demographics_path', ['', 'inf', '1e300']),
        ]:
            original = Path(self.paths[name]).read_bytes()
            for value in values:
                with self.subTest(name=name, value=value):
                    self.rewrite_csv(name, lambda rows: rows[-1].__setitem__(0, value))
                    with self.assertRaises(ValueError):
                        self.build()
                    Path(self.paths[name]).write_bytes(original)

    def test_reserved_test_values_are_not_used_to_construct_train_or_validation(self):
        def change(rows):
            for row in rows[1:]:
                if row[-1] in ['0009', '0010']:
                    row[:-1] = ['nan'] * (len(row) - 1)

        for name in ['event_times_path', 'is_event_path', 'demographics_path']:
            self.rewrite_csv(name, change)
        self.assertEqual(len(self.build()), 6)
        self.assertEqual(len(self.build('validation')), 3)
        with self.assertRaises(ValueError):
            self.build('test')

    def test_hash_snapshot_and_returned_state_cannot_be_changed_silently(self):
        dataset = self.build()
        first = dataset[0]
        first['event_times'].fill_(-1)
        first['demographics'].fill_(99)
        self.assertTrue((dataset[0]['event_times'] >= 0).all())
        self.assertNotEqual(dataset[0]['demographics'][0].item(), 99)
        schema = dataset.schema
        schema['endpoint_names'].reverse()
        self.assertEqual(dataset.endpoint_names[0], 'synthetic_0')
        with self.assertRaises(AttributeError):
            dataset.modality_names = ('RESP', 'BAS')
        selected = copy.copy(dataset)
        selected.info = dataset.info.iloc[[2, 0]].copy()
        self.assertEqual(selected[0]['study_id'], '0002')
        selected.info.loc[selected.info.index[0], 'subject_id'] = 'changed'
        with self.assertRaises(ValueError):
            selected[0]
        path = Path(self.directory.name) / self.manifest['records'][0]['path']
        with path.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            dataset[0]

    def test_schema_manifest_types_and_duplicate_json_fields(self):
        original = copy.deepcopy(self.manifest)
        changes = [
            lambda m: m.__setitem__('version', True),
            lambda m: m['schema'].__setitem__('embedding_width', 127),
            lambda m: m['schema'].__setitem__('embedding_step_seconds', 5.0),
            lambda m: m['schema'].__setitem__('data_kind', 'unknown'),
            lambda m: m['schema'].__setitem__('endpoint_names', ['one'] * 1065),
            lambda m: m['schema'].__setitem__('modality_names', ['EEG']),
            lambda m: m['schema'].__setitem__('upstream_checkpoint_sha256', 'bad'),
            lambda m: m['records'][0].__setitem__('study_id', 0),
            lambda m: m['records'][0].__setitem__('split', 'unknown'),
        ]
        for change in changes:
            manifest = copy.deepcopy(original)
            change(manifest)
            save_manifest(self.paths, manifest)
            with self.assertRaises(ValueError):
                self.build()
        path = Path(self.paths['manifest_path'])
        path.write_text('{"version":1,"version":1}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            self.build()
        save_manifest(self.paths, original)

    def test_hdf5_shapes_values_and_full_window_limits(self):
        cases = [
            lambda f: f.create_dataset(
                'RESP', data=np.zeros((2, 128), dtype='float32')
            ),
            lambda f: f.create_dataset('BAS', data=np.zeros((2, 129), dtype='float32')),
            lambda f: f.create_dataset('BAS', data=np.zeros((0, 128), dtype='float32')),
            lambda f: f.create_dataset(
                'BAS', data=np.zeros((6481, 128), dtype='float32')
            ),
            lambda f: f.create_dataset('BAS', data=np.zeros((2, 128), dtype='float64')),
            lambda f: f.create_dataset('BAS', data=np.zeros((2, 128), dtype='int32')),
            lambda f: f.create_dataset(
                'BAS', data=np.full((2, 128), np.nan, dtype='float32')
            ),
            lambda f: f.create_group('BAS'),
        ]
        for writer in cases:
            replace_embedding(self.paths, self.manifest, 0, writer)
            with self.assertRaises(ValueError):
                self.build()[0]

        def unequal(f):
            f.create_dataset('BAS', data=np.zeros((2, 128), dtype='float32'))
            f.create_dataset('RESP', data=np.zeros((3, 128), dtype='float32'))

        replace_embedding(self.paths, self.manifest, 0, unequal)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            self.build()[0]

        def undeclared(f):
            f.create_dataset('BAS', data=np.zeros((2, 128), dtype='float32'))
            f.create_dataset('other', data=np.zeros((2, 128), dtype='float32'))

        replace_embedding(self.paths, self.manifest, 0, undeclared)
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            self.build()[0]

    def test_self_contained_hdf5_rejects_external_soft_and_virtual_links(self):
        external = Path(self.directory.name) / 'external.h5'
        with h5py.File(external, 'w') as source:
            source.create_dataset('data', data=np.zeros((2, 128), dtype='float32'))

        def linked(f):
            f['BAS'] = h5py.ExternalLink(str(external), 'data')

        replace_embedding(self.paths, self.manifest, 0, linked)
        with self.assertRaisesRegex(ValueError, 'links'):
            self.build()[0]

        def soft(f):
            f.create_dataset('RESP', data=np.zeros((2, 128), dtype='float32'))
            f['BAS'] = h5py.SoftLink('/RESP')

        replace_embedding(self.paths, self.manifest, 0, soft)
        with self.assertRaisesRegex(ValueError, 'links'):
            self.build()[0]

        def virtual(f):
            layout = h5py.VirtualLayout(shape=(2, 128), dtype='float32')
            layout[:] = h5py.VirtualSource(str(external), 'data', shape=(2, 128))
            f.create_virtual_dataset('BAS', layout)

        replace_embedding(self.paths, self.manifest, 0, virtual)
        external.unlink()
        with self.assertRaisesRegex(ValueError, 'self-contained'):
            self.build()[0]

        def external_storage(f):
            f.create_dataset(
                'BAS',
                shape=(2, 128),
                dtype='float32',
                external=[(str(external), 0, h5py.h5f.UNLIMITED)],
            )

        replace_embedding(self.paths, self.manifest, 0, external_storage)
        with self.assertRaisesRegex(ValueError, 'self-contained'):
            self.build()[0]

    def test_default_dtype_does_not_change_reader_or_collator_precision(self):
        previous = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous)
        torch.set_default_dtype(torch.float64)
        dataset = self.build()
        batch = collate_sleepfm_survival([dataset[0], dataset[1]])
        self.assertEqual(batch['embeddings'].dtype, torch.float32)
        self.assertEqual(batch['demographics'].dtype, torch.float32)
        self.assertEqual(batch['event_times'].dtype, torch.float64)

    def test_spawn_loader_reads_each_baseline_once_without_open_file_handles(self):
        dataset = self.build()
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=1,
            multiprocessing_context='spawn',
            collate_fn=collate_sleepfm_survival,
        )
        studies = []
        for batch in loader:
            studies.extend(batch['study_id'])
        self.assertEqual(studies, dataset.info['study_id'].tolist())


if __name__ == '__main__':
    unittest.main()
