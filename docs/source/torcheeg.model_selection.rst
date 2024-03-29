Dataset Splitting
=================================

.. contents:: Extensive dataset partitioning methods for users to experiment with different settings.
    :depth: 2
    :local:
    :backlinks: top

KFold
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.KFold

KFoldPerSubject
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.KFoldPerSubject

KFoldCrossSubject
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.KFoldCrossSubject

KFoldGroupbyTrial
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.KFoldGroupbyTrial

KFoldPerSubjectGroupbyTrial
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.KFoldPerSubjectGroupbyTrial

KFoldCrossTrial
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.KFoldCrossTrial

KFoldPerSubjectCrossTrial
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.KFoldPerSubjectCrossTrial

LeaveOneSubjectOut
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.LeaveOneSubjectOut


Subcategory
-------------------------------------------------

.. autoclass:: torcheeg.model_selection.Subcategory

train_test_split
-----------------------------------------------

.. autofunction:: torcheeg.model_selection.train_test_split

train_test_split_groupby_trial
-----------------------------------------------

.. autofunction:: torcheeg.model_selection.train_test_split_groupby_trial

train_test_split_per_subject_groupby_trial
-----------------------------------------------

.. autofunction:: torcheeg.model_selection.train_test_split_per_subject_groupby_trial

train_test_split_cross_trial
-----------------------------------------------

.. autofunction:: torcheeg.model_selection.train_test_split_cross_trial

train_test_split_per_subject_cross_trial
-----------------------------------------------

.. autofunction:: torcheeg.model_selection.train_test_split_per_subject_cross_trial
