import importlib
import unittest


class TestModelSelectionExports(unittest.TestCase):

    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.model_selection')
        expected = {'BaseModelSelection': 'torcheeg.model_selection.base_model_selection', 'KFold': 'torcheeg.model_selection.k_fold', 'KFoldCrossSubject': 'torcheeg.model_selection.k_fold_cross_subject', 'KFoldCrossSubjectTrainValTest': 'torcheeg.model_selection.k_fold_cross_subject_train_val_test', 'KFoldCrossTrial': 'torcheeg.model_selection.k_fold_cross_trial', 'KFoldGroupbyTrial': 'torcheeg.model_selection.k_fold_groupby_trial', 'KFoldPerSubject': 'torcheeg.model_selection.k_fold_per_subject', 'KFoldPerSubjectCrossTrial': 'torcheeg.model_selection.k_fold_per_subject_cross_trial', 'KFoldPerSubjectGroupbyTrial': 'torcheeg.model_selection.k_fold_per_subject_groupby_trial', 'LeaveOneSubjectOut': 'torcheeg.model_selection.leave_one_subject_out', 'Subcategory': 'torcheeg.model_selection.subcategory', 'TrainTestSplitPerSubjectGroupbyTrial':
                    'torcheeg.model_selection.split_per_subject_groupby_trial_class', 'TrainValTestSplitCrossSubject': 'torcheeg.model_selection.train_val_test_split_cross_subject', 'TrainValTestSplitCrossTrial': 'torcheeg.model_selection.train_val_test_split_cross_trial', 'TrainValTestSplitGroupbyFolder': 'torcheeg.model_selection.train_val_test_split_groupby_folder', 'train_test_split': 'torcheeg.model_selection.split', 'train_test_split_cross_subject': 'torcheeg.model_selection.split_cross_subject', 'train_test_split_cross_trial': 'torcheeg.model_selection.split_cross_trial', 'train_test_split_groupby_trial': 'torcheeg.model_selection.split_groupby_trial', 'train_test_split_per_subject_cross_trial': 'torcheeg.model_selection.split_per_subject_cross_trial', 'train_test_split_per_subject_groupby_trial': 'torcheeg.model_selection.split_per_subject_groupby_trial'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == '__main__':
    unittest.main()
